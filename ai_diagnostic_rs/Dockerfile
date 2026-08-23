FROM rust:1.95-slim AS builder

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /build/target/release/ai_diagnostic /usr/local/bin/ai_diagnostic

EXPOSE 8000
ENV HISTORY_DB_PATH=/data/diagnosis_history.db
VOLUME ["/data"]

CMD ["ai_diagnostic", "serve", "--host", "0.0.0.0", "--port", "8000"]
