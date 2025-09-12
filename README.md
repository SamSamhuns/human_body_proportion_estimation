# Person Detection, Pose and Body Proportion Estimation

[![Python 3.10](https://img.shields.io/badge/python-3.10-green.svg)](https://www.python.org/downloads/release/python-3100/)[![Python 3.11](https://img.shields.io/badge/python-3.11-green.svg)](https://www.python.org/downloads/release/python-3110/)

- [Person Detection, Pose and Body Proportion Estimation](#person-detection-pose-and-body-proportion-estimation)
  - [Download model weights](#download-model-weights)
  - [Requirements](#requirements)
  - [1. Build and run docker image for uvicorn server with fastAPI exposed and triton-server in the backend](#1-build-and-run-docker-image-for-uvicorn-server-with-fastapi-exposed-and-triton-server-in-the-backend)
  - [2. Build and run docker image for triton-server only](#2-build-and-run-docker-image-for-triton-server-only)
    - [Run triton-server container and test](#run-triton-server-container-and-test)
  - [CPU mode](#cpu-mode)
  - [Performance Benchmarking](#performance-benchmarking)

## Download model weights

[Google Drive Link](https://drive.google.com/file/d/1W1OLyrOdKrWPSWDNnLW3WfQkwNCnknqA/view?usp=sharing), or use `gdown` to download.

```bash
python3 -m venv venv
source venv/bin/activate
# inside venv/virtualenv/conda
pip install gdown
# download model weights
gdown 1W1OLyrOdKrWPSWDNnLW3WfQkwNCnknqA
unzip models.zip
rm models.zip
```

## Requirements

Tested with [Docker compose](https://docs.docker.com/compose/install/) version `v2.28.1`.

Two options:

1. Run uvicorn server with fastAPI exposed and triton-server in the docker container
2. Run triton-server container only and run server locally outside the container

## 1. Build and run docker image for uvicorn server with fastAPI exposed and triton-server in the backend

```shell
docker compose build uvi_trt_server
bash scripts/run_docker_uvicorn_fastapi_server.sh -h EXPOSED_HTTP_PORT # Wait for model loading (60s)
# check localhost:EXPOSED_HTTP_PORT for fastapi page
```

## 2. Build and run docker image for triton-server only

```shell
docker compose build trt_server
```

Use [poetry](https://python-poetry.org/) to install requirements (Recommended):

```shell
poetry install
```

Or, use [pip](https://pip.pypa.io/en/stable/) to install requirements:

```shell
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run triton-server container and test

```shell
bash scripts/run_docker_triton_server.sh -p 8081 # 8081 is the exposed GRPC port. Wait for model loading (60s)
# test to verify working container
python human_body_length_est/person_det_pose_edet4_trtserver.py
# to start the uvicorn server, default port is 8080
PYTHONPATH="./human_body_length_est" python uvicorn_server/server.py [EXPOSED_HTTP_PORT]
```

## CPU mode

If nvidia device drivers are not available, remove the resources section in `docker-compose.yml` and change the instance group in `config.pbtxt` inside all the models to:

```yaml
instance_group [
    {
      count: 1
      kind: KIND_CPU
    }
  ]
```

## Performance Benchmarking

```shell
docker cp perf_analyzer DOCKER_CONTAINER_NAME:PATH
./perf_analyzer -m MODEL_NAME --percentile=95 -u localhost:8081 -i gRPC -b 1 --shape INPUT_NODE_NAME:1,300,300,3 --input-data random --concurrency-range 5:20:5
```
