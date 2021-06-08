#!/bin/bash
# Always Run Triton on HTTP port 8082, GRPC port 8081 inside docker
tritonserver --model-store /cv_root/models --allow-grpc=true --allow-http=false --grpc-port=8081 --allow-metrics=false --allow-gpu-metrics=false &
P1=$!

# Run Fast API Server on port 8080. Should be 8080 always
python3 /cv_root/server.py 8080 &
P2=$!
wait $P1 $P2
