FROM ubuntu:22.04 AS build

RUN apt-get update && apt-get install -y \
    build-essential cmake clang git \
    libssl-dev libcurl4-openssl-dev ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Build AVX2 release binary
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_AVX2=ON
RUN cmake --build build -j"$(nproc)"

FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    libssl3 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# copy built binary + symlink
COPY --from=build /app/build/ndd-avx2 /usr/local/bin/ndd-avx2
COPY --from=build /app/build/ndd /usr/local/bin/ndd

EXPOSE 8080
ENV NDD_PORT=8080
CMD ["ndd"]
