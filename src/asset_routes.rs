//! Routes whose responses carry item assets, behind one shared concurrency limit.
//!
//! A large search builds tens of megabytes of JSON, so a handful at once is enough to exhaust
//! the process memory. Requests past the limit wait for a slot instead of running together.
//! A slot is held until the response body has been sent or dropped, so the bodies of finished
//! searches waiting on slow clients count against the limit too. Each slot is held at most a
//! fixed deadline, building and sending together: a response not sent by then is cut off and its
//! slot given back, so clients that read slowly cannot hold every slot. A request that waits
//! longer than the wait limit for a slot is refused with 503 and `Retry-After`.
//! Routes outside this router (landing, collections, KML, health) are not limited.

use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::task::{Context, Poll};
use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::extract::Request;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::middleware::{self, Next};
use axum::{routing::MethodRouter, Router};
use http_body::{Frame, SizeHint};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::task::AbortHandle;
use tokio::time::Instant;
use tracing::warn;

/// Environment variable naming how many asset-bearing requests may run at once.
pub const CONCURRENCY_VAR: &str = "STAC_ASSET_CONCURRENCY";

/// Sized to a 2 GiB memory limit: the process idles at about 190 MiB and one `limit=300` search
/// (a 33 MB body) adds about 60 MiB while it is built and sent. Measured in a 2 GiB cgroup with
/// this default, 20 parallel `limit=300` searches peaked at 460 to 570 MiB and 100 at 730 MiB.
pub const DEFAULT_CONCURRENCY: usize = 4;

/// Environment variable naming the seconds a request may hold its slot, building and sending
/// its response together.
pub const DEADLINE_VAR: &str = "STAC_RESPONSE_DEADLINE";

/// Long enough to send a `limit=300` search (33 MB) at about 4.5 Mbit/s.
pub const DEFAULT_DEADLINE_SECS: u64 = 60;

/// Environment variable naming the seconds a request may wait for a slot before it is refused.
pub const WAIT_VAR: &str = "STAC_SLOT_WAIT";

pub const DEFAULT_WAIT_SECS: u64 = 30;

/// The limit from the environment value; a missing, non-numeric or zero value takes the default.
pub fn concurrency(value: Option<&str>) -> usize {
    positive(CONCURRENCY_VAR, value, DEFAULT_CONCURRENCY as u64) as usize
}

/// A positive whole number from an environment value, or the default when it is missing or not one.
fn positive(name: &str, value: Option<&str>, default: u64) -> u64 {
    let Some(raw) = value.map(str::trim).filter(|v| !v.is_empty()) else {
        return default;
    };
    match raw.parse::<u64>() {
        Ok(n) if n > 0 => n,
        _ => {
            warn!("{name}={raw:?} is not a positive whole number: using {default}");
            default
        }
    }
}

/// How many requests hold a slot at once, how long one may hold it, and how long one may wait.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Limits {
    pub concurrency: usize,
    pub deadline: Duration,
    pub wait: Duration,
}

impl Limits {
    /// The limits from the environment, read through `var`; each value missing or invalid takes its default.
    pub fn from_env(var: impl Fn(&str) -> Option<String>) -> Self {
        let secs = |name, default| Duration::from_secs(positive(name, var(name).as_deref(), default));
        Limits {
            concurrency: concurrency(var(CONCURRENCY_VAR).as_deref()),
            deadline: secs(DEADLINE_VAR, DEFAULT_DEADLINE_SECS),
            wait: secs(WAIT_VAR, DEFAULT_WAIT_SECS),
        }
    }
}

/// The search, collection items and item routes, sharing one limit of requests in flight.
pub fn router<S: Clone + Send + Sync + 'static>(
    search: MethodRouter<S>,
    collection_items: MethodRouter<S>,
    item: MethodRouter<S>,
    limits: Limits,
) -> Router<S> {
    // One semaphore shared across every route and method below; a per-route limit would give
    // each route its own allowance.
    let slots = Arc::new(Semaphore::new(limits.concurrency));
    Router::new()
        .route("/stac/search", search)
        .route("/stac/collections/:collection_id/items", collection_items)
        .route("/stac/collections/:collection_id/items/:item_id", item)
        .layer(middleware::from_fn(move |req: Request, next: Next| {
            let slots = slots.clone();
            async move {
                let Ok(slot) = tokio::time::timeout(limits.wait, slots.acquire_owned()).await else {
                    warn!("A request waited {}s for a slot and was refused", limits.wait.as_secs_f32());
                    return busy(limits.wait);
                };
                let slot = slot.expect("the slot semaphore is never closed");
                let deadline = Instant::now() + limits.deadline;
                match tokio::time::timeout_at(deadline, next.run(req)).await {
                    Ok(response) => response.map(|inner| Body::new(SlotBody::new(inner, slot, deadline))),
                    Err(_) => {
                        warn!("A response was not built within {}s and was refused", limits.deadline.as_secs_f32());
                        busy(limits.wait)
                    }
                }
            }
        }))
}

/// 503 with a `Retry-After` of the wait limit, in whole seconds and at least one.
fn busy(wait: Duration) -> Response {
    let retry_after = wait.as_secs_f64().ceil().max(1.0) as u64;
    (StatusCode::SERVICE_UNAVAILABLE, [(header::RETRY_AFTER, retry_after.to_string())], "Too many requests in flight")
        .into_response()
}

/// A response body that holds its slot until it has been sent, dropped, or its deadline passes.
///
/// A client that stops reading leaves the body unpolled, so the deadline is kept by a timer task
/// rather than by polling: at the deadline it drops the body and the slot together, and the next
/// poll ends the response with an error, which closes the connection.
struct SlotBody {
    held: Arc<Mutex<Option<Held>>>,
    finished: bool,
    expiry: AbortHandle,
}

struct Held {
    inner: Body,
    _slot: OwnedSemaphorePermit,
}

impl SlotBody {
    fn new(inner: Body, slot: OwnedSemaphorePermit, deadline: Instant) -> Self {
        let held = Arc::new(Mutex::new(Some(Held { inner, _slot: slot })));
        let expiring = Arc::downgrade(&held);
        let expiry = tokio::spawn(async move {
            tokio::time::sleep_until(deadline).await;
            if let Some(held) = expiring.upgrade() {
                let expired = lock(&held).take();
                if expired.is_some() {
                    warn!("A response was not sent before its deadline and was cut off");
                }
            }
        })
        .abort_handle();
        SlotBody { held, finished: false, expiry }
    }
}

fn lock(held: &Mutex<Option<Held>>) -> MutexGuard<'_, Option<Held>> {
    held.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Drop for SlotBody {
    fn drop(&mut self) {
        self.expiry.abort();
    }
}

impl http_body::Body for SlotBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, axum::Error>>> {
        if self.finished {
            return Poll::Ready(None);
        }
        let mut held = lock(&self.held);
        let Some(Held { inner, .. }) = held.as_mut() else {
            return Poll::Ready(Some(Err(axum::Error::new("the response was not sent before its deadline"))));
        };
        let frame = Pin::new(inner).poll_frame(cx);
        if let Poll::Ready(None) = frame {
            // Sent in full: the slot is given back now rather than when the body is dropped.
            held.take();
            drop(held);
            self.finished = true;
            self.expiry.abort();
        }
        frame
    }

    fn is_end_stream(&self) -> bool {
        self.finished || lock(&self.held).as_ref().is_some_and(|held| held.inner.is_end_stream())
    }

    fn size_hint(&self) -> SizeHint {
        lock(&self.held).as_ref().map_or_else(SizeHint::default, |held| held.inner.size_hint())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{header, Request, StatusCode};
    use axum::routing::get;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::sync::Notify;
    use tower::ServiceExt;

    #[derive(Clone, Default)]
    struct Gauge {
        now: Arc<AtomicUsize>,
        max: Arc<AtomicUsize>,
        started: Arc<AtomicUsize>,
        release: Arc<Notify>,
        hold: bool,
    }

    impl Gauge {
        async fn run(&self) -> &'static str {
            let n = self.now.fetch_add(1, Ordering::SeqCst) + 1;
            self.max.fetch_max(n, Ordering::SeqCst);
            self.started.fetch_add(1, Ordering::SeqCst);
            if self.hold {
                self.release.notified().await;
            } else {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            self.now.fetch_sub(1, Ordering::SeqCst);
            "ok"
        }
    }

    fn app(gauge: &Gauge, limit: usize) -> Router {
        let limits =
            Limits { concurrency: limit, deadline: Duration::from_secs(60), wait: Duration::from_secs(60) };
        app_with(gauge, limits)
    }

    fn app_with(gauge: &Gauge, limits: Limits) -> Router {
        let handler = |g: Gauge| move || async move { g.run().await };
        router(
            get(handler(gauge.clone())).post(handler(gauge.clone())),
            get(handler(gauge.clone())),
            get(handler(gauge.clone())),
            limits,
        )
        .route("/health", get(|| async { "healthy" }))
    }

    async fn status(app: &Router, method: &str, path: &str) -> StatusCode {
        let req = Request::builder().method(method).uri(path).body(Body::empty()).unwrap();
        app.clone().oneshot(req).await.unwrap().status()
    }

    #[tokio::test]
    async fn requests_past_the_limit_wait_and_all_answer() {
        let gauge = Gauge::default();
        let app = app(&gauge, 3);
        let requests = (0..24).map(|i| {
            let app = app.clone();
            let (method, path) = match i % 4 {
                0 => ("GET", "/stac/search"),
                1 => ("POST", "/stac/search"),
                2 => ("GET", "/stac/collections/c/items"),
                _ => ("GET", "/stac/collections/c/items/i"),
            };
            tokio::spawn(async move { status(&app, method, path).await })
        });
        for handle in requests.collect::<Vec<_>>() {
            assert_eq!(handle.await.unwrap(), StatusCode::OK);
        }
        assert_eq!(gauge.started.load(Ordering::SeqCst), 24);
        assert_eq!(gauge.max.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn health_answers_while_every_slot_is_taken() {
        let gauge = Gauge { hold: true, ..Gauge::default() };
        let app = app(&gauge, 2);
        let held: Vec<_> = (0..4)
            .map(|_| {
                let app = app.clone();
                tokio::spawn(async move { status(&app, "GET", "/stac/search").await })
            })
            .collect();
        while gauge.started.load(Ordering::SeqCst) < 2 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(gauge.started.load(Ordering::SeqCst), 2, "a request ran past the limit");

        let health = tokio::time::timeout(Duration::from_secs(1), status(&app, "GET", "/health")).await;
        assert_eq!(health.expect("/health waited for a search slot"), StatusCode::OK);

        while gauge.started.load(Ordering::SeqCst) < 4 || gauge.now.load(Ordering::SeqCst) > 0 {
            gauge.release.notify_waiters();
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        for handle in held {
            assert_eq!(handle.await.unwrap(), StatusCode::OK);
        }
    }

    #[tokio::test]
    async fn a_slot_stays_taken_until_the_response_body_is_dropped() {
        let gauge = Gauge::default();
        let app = app(&gauge, 1);
        let req = || Request::builder().uri("/stac/search").body(Body::empty()).unwrap();
        let unsent = app.clone().oneshot(req()).await.unwrap();
        assert_eq!(unsent.status(), StatusCode::OK);

        let next = tokio::spawn({
            let app = app.clone();
            async move { status(&app, "GET", "/stac/search").await }
        });
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(gauge.started.load(Ordering::SeqCst), 1, "a request ran while a body was unsent");

        drop(unsent);
        let answered = tokio::time::timeout(Duration::from_secs(1), next).await;
        assert_eq!(answered.expect("the slot was not given back").unwrap(), StatusCode::OK);
        assert_eq!(gauge.started.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn a_slot_is_given_back_at_the_deadline_to_clients_that_never_read() {
        let gauge = Gauge::default();
        let limits =
            Limits { concurrency: 4, deadline: Duration::from_millis(200), wait: Duration::from_secs(5) };
        let app = app_with(&gauge, limits);
        let req = || Request::builder().uri("/stac/search").body(Body::empty()).unwrap();
        let mut unread = Vec::new();
        for _ in 0..4 {
            let response = app.clone().oneshot(req()).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            unread.push(response);
        }

        let fifth = tokio::time::timeout(Duration::from_secs(2), status(&app, "GET", "/stac/search")).await;
        assert_eq!(fifth.expect("no slot was given back at the deadline"), StatusCode::OK);
        assert_eq!(gauge.started.load(Ordering::SeqCst), 5);

        tokio::time::sleep(Duration::from_millis(200)).await;
        for response in unread {
            let cut = axum::body::to_bytes(response.into_body(), usize::MAX).await;
            assert!(cut.is_err(), "a response past its deadline was still sent");
        }
    }

    #[tokio::test]
    async fn a_response_still_building_at_the_deadline_is_refused_and_its_slot_given_back() {
        let gauge = Gauge { hold: true, ..Gauge::default() };
        let limits =
            Limits { concurrency: 1, deadline: Duration::from_millis(100), wait: Duration::from_secs(5) };
        let app = app_with(&gauge, limits);
        let response = tokio::time::timeout(
            Duration::from_secs(2),
            app.clone().oneshot(Request::builder().uri("/stac/search").body(Body::empty()).unwrap()),
        )
        .await
        .expect("a request ran past its deadline")
        .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(response.headers().contains_key(header::RETRY_AFTER));

        let next = tokio::spawn({
            let app = app.clone();
            async move { status(&app, "GET", "/stac/search").await }
        });
        while gauge.started.load(Ordering::SeqCst) < 2 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        gauge.release.notify_waiters();
        assert_eq!(next.await.unwrap(), StatusCode::OK);
    }

    #[tokio::test]
    async fn a_request_waiting_past_the_wait_limit_is_told_to_retry() {
        let gauge = Gauge::default();
        let limits =
            Limits { concurrency: 1, deadline: Duration::from_secs(60), wait: Duration::from_millis(100) };
        let app = app_with(&gauge, limits);
        let req = || Request::builder().uri("/stac/search").body(Body::empty()).unwrap();
        let _unread = app.clone().oneshot(req()).await.unwrap();

        let queued = tokio::time::timeout(Duration::from_secs(2), app.clone().oneshot(req()))
            .await
            .expect("a queued request waited past the wait limit")
            .unwrap();
        assert_eq!(queued.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(queued.headers()[header::RETRY_AFTER], "1");
        assert_eq!(gauge.started.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_body_read_to_the_end_gives_its_slot_back_before_it_is_dropped() {
        let gauge = Gauge::default();
        let app = app(&gauge, 1);
        let req = || Request::builder().uri("/stac/search").body(Body::empty()).unwrap();
        let mut body = app.clone().oneshot(req()).await.unwrap().into_body();
        while let Some(frame) =
            std::future::poll_fn(|cx| http_body::Body::poll_frame(Pin::new(&mut body), cx)).await
        {
            frame.unwrap();
        }
        let next = tokio::time::timeout(Duration::from_secs(1), status(&app, "GET", "/stac/search")).await;
        assert_eq!(next.expect("a finished body kept its slot"), StatusCode::OK);
        drop(body);
    }

    #[test]
    fn limits_read_positive_numbers_or_take_the_defaults() {
        let env = |pairs: &'static [(&'static str, &'static str)]| {
            move |name: &str| pairs.iter().find(|(k, _)| *k == name).map(|(_, v)| v.to_string())
        };
        assert_eq!(
            Limits::from_env(env(&[])),
            Limits {
                concurrency: DEFAULT_CONCURRENCY,
                deadline: Duration::from_secs(DEFAULT_DEADLINE_SECS),
                wait: Duration::from_secs(DEFAULT_WAIT_SECS),
            }
        );
        let set = Limits::from_env(env(&[(CONCURRENCY_VAR, "6"), (DEADLINE_VAR, "90"), (WAIT_VAR, " 5 ")]));
        assert_eq!(set, Limits { concurrency: 6, deadline: Duration::from_secs(90), wait: Duration::from_secs(5) });
        let bad = Limits::from_env(env(&[(DEADLINE_VAR, "0"), (WAIT_VAR, "soon")]));
        assert_eq!(bad.deadline, Duration::from_secs(DEFAULT_DEADLINE_SECS));
        assert_eq!(bad.wait, Duration::from_secs(DEFAULT_WAIT_SECS));
    }

    #[test]
    fn concurrency_takes_a_positive_number_or_the_default() {
        assert_eq!(concurrency(None), DEFAULT_CONCURRENCY);
        assert_eq!(concurrency(Some("")), DEFAULT_CONCURRENCY);
        assert_eq!(concurrency(Some(" 6 ")), 6);
        assert_eq!(concurrency(Some("0")), DEFAULT_CONCURRENCY);
        assert_eq!(concurrency(Some("-2")), DEFAULT_CONCURRENCY);
        assert_eq!(concurrency(Some("four")), DEFAULT_CONCURRENCY);
    }
}
