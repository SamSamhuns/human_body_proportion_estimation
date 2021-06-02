# Person Detection, Pose and Body Proportion Estimation

## Build and Run Docker image

```shell
# set a virtualenv/conda env
$ pip install -r requirements.txt
$ pip install nvidia-pyindex==1.0.6
$ pip install tritonclient[grpc]==2.6.0
$ bash build_run_docker.sh -h <EXPOSED_HTTP_PORT_NUM>
# Wait for model loading (30-60 seconds approx).
```
