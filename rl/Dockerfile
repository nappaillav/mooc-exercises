#ARG DOCKER_REGISTRY

#ARG ARCH=amd64
#ARG MAJOR=daffy
#ARG BASE_TAG=${MAJOR}-${ARCH}

#FROM ${DOCKER_REGISTRY}/duckietown/challenge-aido_lf-template-pytorch:${BASE_TAG}
# ${DOCKER_REGISTRY}/duckietown/challenge-aido_lf-baseline-duckietown-ml:${BASE_TAG}}

# let's create our workspace, we don't want to clutter the container
#RUN rm -rf /workspace; mkdir /workspace


# we make the workspace our working directory
#WORKDIR /workspace

#ARG PIP_INDEX_URL="https://pypi.org/simple"
#ENV PIP_INDEX_URL=${PIP_INDEX_URL}


# here, we install the requirements, some requirements come by default
# you can add more if you need to in requirements.txt
#COPY requirements.* ./
#RUN cat requirements.* > .requirements.txt
#RUN python3 -m pip install  -r .requirements.txt

# Juuuuust in case...
#RUN python3 -m pip uninstall dataclasses -y

# let's copy all our solution files to our workspace
# if you have more file use the COPY command to move them to the workspace
#COPY /solution/*.py ./
#COPY /solution/models /workspace/models
#COPY node_launch.yaml .

#RUN node-launch --config node_launch.yaml --check

#ENTRYPOINT ["node-launch", "--config", "node_launch.yaml"]

# Definition of Submission container
ARG ARG DOCKER_REGISTRY=docker.io
ARG ARCH="amd64"
ARG MAJOR=daffy
ARG BASE_TAG=${MAJOR}-${ARCH}
#FROM ${DOCKER_REGISTRY}/duckietown/dt-machine-learning-base-environment:${BASE_TAG}

#FROM ${DOCKER_REGISTRY}/duckietown/challenge-aido_lf-baseline-duckietown-ml:${BASE_TAG}
FROM ${DOCKER_REGISTRY}/duckietown/challenge-aido_lf-template-pytorch:${BASE_TAG}

# DO NOT MODIFY: your submission won't run if you do
RUN apt-get update -y && apt-get install -y --no-install-recommends \
         gcc \
         libc-dev\
         git \
         bzip2 \
         python-tk && \
     rm -rf /var/lib/apt/lists/*

# let's create our workspace, we don't want to clutter the container
# RUN rm -rf /agent
# RUN mkdir /agent
WORKDIR /agent
# WORKDIR /agent

ARG PIP_INDEX_URL
ENV PIP_INDEX_URL=${PIP_INDEX_URL}

# here, we install the requirements, some requirements come by default
# you can add more if you need to in requirements.txt
COPY requirements.* ./
RUN cat requirements.txt > .requirements.txt
RUN echo PIP_INDEX_URL=$PIP_INDEX_URL
RUN python3 -m pip install -r .requirements.txt
RUN python3 -m pip uninstall -y dataclasses


COPY solution /agent/solution

# we make the workspace our working directory


# RUN python3 -c "import solution.agent; import wrappers; import model; import helpers;"

# CMD python3 solution.py
CMD ["python3",  "/agent/solution/agent.py"]

# RUN node-launch --config node_launch.yaml --check

# ENTRYPOINT ["node-launch", "--config", "node_launch.yaml"]
