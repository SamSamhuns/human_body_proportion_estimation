# Person Detection, Pose and Body Proportion Estimation

Important: `docker-compose` requires version `v1.28.0+`. Install with `pip install docker-compose` (inside a virtual env) to get latest version if it not updated.

## Build and Run Docker image for uvicorn server with fastAPI exposed and triton-server in the backend

```shell
$ docker-compose build uvi_trt_server
$ bash run_docker_uvicorn_fastapi_server.sh -g EXPOSED_HTTP_PORT # Wait for model loading (60s)
# check localhost:EXPOSED_HTTP_PORT for fastapi page
```

## Build and Run Docker image for triton-server only

```shell
# set a python virtual env/conda env
$ pip install Pillow==8.1.0
$ pip install opencv-python==4.5.1.48
$ pip install nvidia-pyindex==1.0.6
$ pip install tritonclient[grpc]==2.6.0
$ docker-compose build trt_server
$ bash run_docker_triton_server.sh -g EXPOSED_GRPC_PORT # Wait for model loading (60s)
$ python human_body_length_est/person_det_pose_est_trtserver_demo.py  # test to verify working container
```
