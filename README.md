# Person Detection, Pose and Body Proportion Estimation

## Download model weights

[Manual Google Drive Download Link](https://drive.google.com/file/d/1hc2QtzLkAh3Ui2qT5FkPDwEe21KAwVgU/view?usp=sharing), or use gdown to download.

```bash
python3 -m venv venv
source venv/bin/activate
# inside venv/virtualenv/conda
pip install gdown
# download model weights
gdown 1hc2QtzLkAh3Ui2qT5FkPDwEe21KAwVgU
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

## Build and Run Docker image for uvicorn server with fastAPI exposed and triton-server in the backend

```shell
docker-compose build uvi_trt_server
bash run_docker_uvicorn_fastapi_server.sh -h EXPOSED_HTTP_PORT # Wait for model loading (60s)
# check localhost:EXPOSED_HTTP_PORT for fastapi page
```

## Build and Run Docker image for triton-server only

```shell
# set a python virtual env/conda env
python -m venv venv
source venv/bin/activate
# install dependencies
pip install -r requirements.txt
docker-compose build trt_server
bash run_docker_triton_server.sh -g EXPOSED_GRPC_PORT # Wait for model loading (60s)
python human_body_length_est/person_det_pose_edet4_trtserver.py  # test to verify working container
```

### Performance Benchmarking

```shell
docker cp perf_analyzer DOCKER_CONTAINER_NAME:PATH
./perf_analyzer -m MODEL_NAME --percentile=95 -u localhost:8081 -i gRPC -b 1 --shape INPUT_NODE_NAME:1,300,300,3 --input-data random --concurrency-range 5:20:5
```
