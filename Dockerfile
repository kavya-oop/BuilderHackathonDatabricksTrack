FROM rust:1.85-bookworm AS builder

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
RUN mkdir -p src && printf 'fn main() {}\n' > src/main.rs && cargo build --release --locked
COPY src/ ./src/
COPY generator/src/ ./generator/src/
COPY pgo.profdata ./pgo.profdata
RUN touch src/main.rs && \
    CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-Cprofile-use=/build/pgo.profdata" \
    cargo build --release --locked --target x86_64-unknown-linux-gnu

FROM debian:bookworm-slim
COPY --from=builder /build/target/x86_64-unknown-linux-gnu/release/shopping-sessions /app/shopping-sessions
RUN chmod +x /app/shopping-sessions
USER 1000:1000
WORKDIR /app
CMD ["/app/shopping-sessions"]
