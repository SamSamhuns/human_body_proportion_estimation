#!/bin/bash

# check for 4 cmd args
if [ $# -ne 2 ]
  then
    echo "http port must be specified for triton-server."
		echo "eg. \$ bash build_run_docker.sh -h 8080"
		exit
fi

# get the http port
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--http) http="$2"; shift ;;
        *) echo "Unknown parameter passed: $1";
	exit 1 ;;
    esac
    shift
done

echo "Running docker with exposed uvicorn+fastapi server http port: $http"
docker-compose run -d \
              -p $http:8080 \
              --name body_est_uvi_trt \
              uvi_trt_server
