//! Routes whose responses carry item assets, behind one shared concurrency limit.
//!
//! A large search builds tens of megabytes of JSON, so a handful at once is enough to exhaust
//! the process memory. Requests past the limit wait for a slot instead of running together.
//! A slot is held until the response body has been sent or dropped, so the bodies of finished
//! searches waiting on slow clients count against the limit too.
//! Routes outside this router (landing, collections, KML, health) are not limited.

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use axum::body::{Body, Bytes};
use axum::extract::Request;
use axum::middleware::{self, Next};
use axum::{routing::MethodRouter, Router};
use http_body::{Frame, SizeHint};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tracing::warn;

/// Environment variable naming how many asset-bearing requests may run at once.
pub const CONCURRENCY_VAR: &str = "STAC_ASSET_CONCURRENCY";

/// Sized to a 2 GiB memory limit: the process idles at about 190 MiB and one `limit=300` search
/// (a 33 MB body) adds about 60 MiB while it is built and sent. Measured in a 2 GiB cgroup with
/// this default, 20 parallel `limit=300` searches peaked at 460 to 570 MiB and 100 at 730 MiB.
pub const DEFAULT_CONCURRENCY: usize = 4;

/// The limit from the environment value; a missing, non-numeric or zero value takes the default.
pub fn concurrency(value: Option<&str>) -> usize {
    let Some(raw) = value.map(str::trim).filter(|v| !v.is_empty()) else {
        return DEFAULT_CONCURRENCY;
    };
    match raw.parse::<usize>() {
        Ok(n) if n > 0 => n,
        _ => {
            warn!("{CONCURRENCY_VAR}={raw:?} is not a positive whole number: using {DEFAULT_CONCURRENCY}");
            DEFAULT_CONCURRENCY
        }
    }
}

/// The search, collection items and item routes, sharing one limit of `limit` requests in flight.
pub fn router<S: Clone + Send + Sync + 'static>(
    search: MethodRouter<S>,
    collection_items: MethodRouter<S>,
    item: MethodRouter<S>,
    limit: usize,
) -> Router<S> {
    // One semaphore shared across every route and method below; a per-route limit would give
    // each route its own allowance.
    let slots = Arc::new(Semaphore::new(limit));
    Router::new()
        .route("/stac/search", search)
        .route("/stac/collections/:collection_id/items", collection_items)
        .route("/stac/collections/:collection_id/items/:item_id", item)
        .layer(middleware::from_fn(move |req: Request, next: Next| {
            let slots = slots.clone();
            async move {
                let slot = slots.acquire_owned().await.expect("the slot semaphore is never closed");
                next.run(req).await.map(|inner| Body::new(SlotBody { inner, _slot: slot }))
            }
        }))
}

/// A response body that gives its slot back when it is dropped, after it has been sent.
struct SlotBody {
    inner: Body,
    _slot: OwnedSemaphorePermit,
}

impl http_body::Body for SlotBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, axum::Error>>> {
        Pin::new(&mut self.inner).poll_frame(cx)
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
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
        let handler = |g: Gauge| move || async move { g.run().await };
        router(
            get(handler(gauge.clone())).post(handler(gauge.clone())),
            get(handler(gauge.clone())),
            get(handler(gauge.clone())),
            limit,
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
