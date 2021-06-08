FROM nvcr.io/nvidia/tritonserver:20.12-py3

ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV CUDA_VISIBLE_DEVICES=0

# install opencv reqs
RUN apt-get update \
 && apt-get install ffmpeg libsm6 libxext6 -y

WORKDIR /cv_root

RUN /usr/bin/python3 -m pip install --upgrade pip
RUN pip install nvidia-pyindex==1.0.6
RUN pip install tritonclient[grpc]==2.6.0
COPY ./uvicorn_server/requirements.txt /cv_root
RUN pip install -r requirements.txt

# copy all model files into docker container
RUN mkdir -p /cv_root/models
COPY models /cv_root/models

# for running fast api with triton server
COPY ./uvicorn_server/start_servers.sh /cv_root
COPY ./uvicorn_server/server.py /cv_root
RUN mkdir -p /cv_root/human_body_length_est
COPY human_body_length_est /cv_root/

RUN ["chmod", "u+x", "/cv_root/start_servers.sh"]
CMD ["/bin/bash", "/cv_root/start_servers.sh"]
