# Build stage
FROM rust:1.90-bookworm AS builder

# Install build dependencies for GDAL bindings
RUN apt-get update && apt-get install -y \
    libgdal-dev \
    libclang-dev \
    clang \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy source
COPY Cargo.toml Cargo.lock ./
COPY src ./src

# Build for release
RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libgdal32 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binary (name from Cargo.toml [[bin]] section)
COPY --from=builder /app/target/release/blatten-api /app/blatten-api

# Copy STAC catalog data
COPY stac /app/stac

# Environment
ENV RUST_LOG=info
ENV STAC_CATALOG_DIR=/app/stac
ENV PORT=3000

EXPOSE 3000

CMD ["/app/blatten-api"]
