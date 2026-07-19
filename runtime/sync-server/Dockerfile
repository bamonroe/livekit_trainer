FROM rust:1.88-bookworm AS build

WORKDIR /app
COPY Cargo.toml Cargo.lock* ./
COPY src ./src
RUN cargo build --release

FROM debian:bookworm AS runtime

# The docker CLI so the server can launch training as a sibling trainer
# container over the mounted host docker socket.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl gnupg \
 && install -m 0755 -d /etc/apt/keyrings \
 && curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc \
 && chmod a+r /etc/apt/keyrings/docker.asc \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian bookworm stable" > /etc/apt/sources.list.d/docker.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends docker-ce-cli \
 && rm -rf /var/lib/apt/lists/*

COPY --from=build /app/target/release/livekit-trainer-sync-server /usr/local/bin/livekit-trainer-sync-server

ENV BIND_ADDR=0.0.0.0:8765
ENV DATA_ROOT=/data/real
ENV INCOMING_ROOT=/incoming/bundles

EXPOSE 8765
CMD ["livekit-trainer-sync-server"]
