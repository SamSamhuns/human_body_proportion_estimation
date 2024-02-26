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

# remove cache
RUN apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR ${WORKDIR}

# setup virtual env for python
ENV VIRTUAL_ENV="${WORKDIR}/venv"
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# install python dependencies & libs
RUN pip install pip==24.0
RUN pip install wheel
COPY ./requirements.txt "$WORKDIR/requirements.txt"
RUN pip install --no-cache-dir --default-timeout=100 -r "$WORKDIR/requirements.txt"

# copy all model files into docker container
RUN mkdir -p "${WORKDIR}/models"
COPY models "${WORKDIR}/models"

# for running fast api with triton server
COPY ./uvicorn_server/start_servers.sh "${WORKDIR}/"
COPY ./uvicorn_server/server.py "${WORKDIR}/"
RUN mkdir -p "${WORKDIR}/human_body_length_est"
COPY human_body_length_est "${WORKDIR}/"

RUN chmod u+x "${WORKDIR}/start_servers.sh"
CMD /bin/bash "${WORKDIR}/start_servers.sh"
