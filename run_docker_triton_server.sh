#!/bin/bash

def_cont_name=body_est_trt

# check for 4 cmd args
if [ $# -ne 2 ]
  then
    echo "GRPC port must be specified for triton-server."
		echo "eg. \$ bash build_run_docker.sh -p 8080"
		exit
fi

# get the grpc port
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -p|--grpc) grpc="$2"; shift ;;
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

echo "Running docker with exposed triton-server grpc port: $grpc"
docker-compose run -d --rm \
              -p $grpc:8081 \
              --name "$def_cont_name" \
              trt_server
