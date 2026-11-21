FROM rust:1.85-bookworm AS builder

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
RUN mkdir -p src && printf 'fn main() {}\n' > src/main.rs && cargo build --release --locked
COPY src/ ./src/
COPY generator/src/ ./generator/src/
RUN touch src/main.rs && cargo build --release --locked

FROM debian:bookworm-slim
COPY --from=builder /build/target/release/shopping-sessions /app/shopping-sessions
RUN chmod +x /app/shopping-sessions
USER 1000:1000
WORKDIR /app
CMD ["/app/shopping-sessions"]
