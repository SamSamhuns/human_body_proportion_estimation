#!/bin/bash

def_cont_name=body_est_uvi_trt

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

# Check if the container is running
if [ "$(docker ps -q -f name=$def_cont_name)" ]; then
    echo "Stopping docker container '$def_cont_name'"
    docker stop "$def_cont_name"
    docker rm "$def_cont_name"
    echo "Stopped & removed container '$def_cont_name'"
fi

echo "Running docker with exposed uvicorn+fastapi server http port: $http"
docker-compose run -d --rm \
              -p $http:8080 \
              --name "$def_cont_name" \
              uvi_trt_server
