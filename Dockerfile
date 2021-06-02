FROM nvcr.io/nvidia/tritonserver:20.12-py3

ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV CUDA_VISIBLE_DEVICES=0

# install opencv reqs
RUN apt-get update \
 && apt-get install ffmpeg libsm6 libxext6 -y

WORKDIR /cv_root
RUN mkdir -p /cv_root/models

RUN pip install nvidia-pyindex==1.0.6
RUN pip install tritonclient[grpc]==2.6.0
COPY perf_analyzer /cv_root/

COPY models /cv_root/models

CMD ["tritonserver", "--model-repository", "/cv_root/models", "--allow-grpc=true", "--allow-http=false", "--grpc-port=8081"]
