# Person Detection, Pose and Body Proportion Estimation

[![Python 3.9](https://img.shields.io/badge/python-3.9-green.svg)](https://www.python.org/downloads/release/python-390/)

- [Person Detection, Pose and Body Proportion Estimation](#person-detection-pose-and-body-proportion-estimation)
  - [Download model weights](#download-model-weights)
  - [Requirements](#requirements)
  - [Build and run docker image for uvicorn server with fastAPI exposed and triton-server in the backend](#build-and-run-docker-image-for-uvicorn-server-with-fastapi-exposed-and-triton-server-in-the-backend)
  - [Build and run docker image for triton-server only](#build-and-run-docker-image-for-triton-server-only)
  - [CPU mode](#cpu-mode)
  - [Performance Benchmarking](#performance-benchmarking)

## Download model weights

[Manual Google Drive Download Link](https://drive.google.com/file/d/1-pSTw19VAYbAKpPvWuYFNm9E3RwNYAIl/view?usp=sharing), or use gdown to download.

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

Install [Docker compose](https://docs.docker.com/compose/install/) if not already present (Recommended). 

It can also be installed inside a python venv.

```shell
pip install docker-compose==1.29.2
pip install docker==6.1.3
```

## Build and run docker image for uvicorn server with fastAPI exposed and triton-server in the backend

```shell
docker-compose build uvi_trt_server
bash run_docker_uvicorn_fastapi_server.sh -h EXPOSED_HTTP_PORT # Wait for model loading (60s)
# check localhost:EXPOSED_HTTP_PORT for fastapi page
```

## Build and run docker image for triton-server only

```shell
# set a python virtual env/conda env
python -m venv venv
source venv/bin/activate
# install dependencies
pip install -r requirements.txt
# build triton-server container
docker-compose build trt_server
```

**Run triton-server container and test**

```shell
bash run_docker_triton_server.sh -p 8081 # 8081 is the exposed GRPC port. Wait for model loading (60s)
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
