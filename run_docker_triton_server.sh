#!/bin/bash

# check for 4 cmd args
if [ $# -ne 2 ]
  then
    echo "GRPC port must be specified for triton-server."
		echo "eg. \$ bash build_run_docker.sh -g 8080"
		exit
fi

# get the grpc port
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -g|--grpc) grpc="$2"; shift ;;
        *) echo "Unknown parameter passed: $1";
	exit 1 ;;
    esac
    shift
done

echo "Running docker with exposed triton-server grpc port: $grpc"
docker-compose run -d \
              -p $grpc:8081 \
              --name body_est_trt \
              trt_server
