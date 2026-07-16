FROM rust:1.88-bookworm AS build

WORKDIR /app
COPY Cargo.toml Cargo.lock* ./
COPY src ./src
RUN cargo build --release

FROM debian:bookworm AS runtime

COPY --from=build /app/target/release/livekit-trainer-sync-server /usr/local/bin/livekit-trainer-sync-server

ENV BIND_ADDR=0.0.0.0:8765
ENV DATA_ROOT=/data/real
ENV INCOMING_ROOT=/incoming/bundles

EXPOSE 8765
CMD ["livekit-trainer-sync-server"]
