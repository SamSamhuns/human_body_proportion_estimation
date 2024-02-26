FROM nvcr.io/nvidia/tritonserver:24.01-py3

ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV WORKDIR="/cv_root"

# install opencv reqs
RUN apt-get update -y --no-install-recommends \
    && apt-get install -y --no-install-recommends python3.10-venv ffmpeg libsm6 libxext6 vim \
    && apt-get clean

WORKDIR ${WORKDIR}

# setup virtual env for python
ENV VIRTUAL_ENV="${WORKDIR}/venv"
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# install python dependencies
RUN pip install wheel==0.42.0 \
    nvidia-pyindex==1.0.9 \
    tritonclient[grpc]==2.42.0

# copy all model files into docker container
RUN mkdir -p "${WORKDIR}/models"
COPY models "${WORKDIR}/models"

# for only running triton server with grpc port
CMD ["tritonserver", "--model-repository", "/cv_root/models", "--allow-grpc=true", "--allow-http=false", "--grpc-port=8081"]
