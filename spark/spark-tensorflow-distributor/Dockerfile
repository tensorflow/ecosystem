ARG UBUNTU_VERSION

FROM ubuntu:${UBUNTU_VERSION}

ARG PYTHON_INSTALL_VERSION

RUN apt-get update && \
    apt install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y wget

RUN apt-get install -y python${PYTHON_INSTALL_VERSION} python${PYTHON_INSTALL_VERSION}-dev python${PYTHON_INSTALL_VERSION}-distutils && \
    apt-get clean && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python$PYTHON_INSTALL_VERSION get-pip.py && \
    rm get-pip.py && \
    ln -s /usr/bin/python${PYTHON_INSTALL_VERSION} /usr/local/bin/python3 && \
    ln -s /usr/bin/python${PYTHON_INSTALL_VERSION} /usr/local/bin/python

RUN apt-get update && \
    apt-get install -y unzip bzip2 build-essential openjdk-8-jdk ssh sudo && \
    apt-get clean

# Add ubuntu user and enable password-less sudo
RUN useradd -mU -s /bin/bash -G sudo ubuntu && \
    echo "ubuntu ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Set up SSH. Docker will cache this layer. So the keys end up the same on all containers.
# We still keep the private key for manual SSH during debugging.
RUN ssh-keygen -q -f ~/.ssh/docker -N "" && \
    cp ~/.ssh/docker.pub ~/.ssh/authorized_keys

# Install Spark and update env variables
# Modify this when the next Spark 3 release includes the allGather barrier mode API
RUN pip install \
    https://ml-team-public-read.s3-us-west-2.amazonaws.com/pyspark-3.1.0.dev0-60dd1a690fed62b1d6442cdc8cf3f89ef4304d5a.tar.gz \
    --force-reinstall --upgrade

# Set SPARK_HOME so that tests can easily find the scripts in $SPARK_HOME/sbin
ENV SPARK_HOME /usr/local/lib/python${PYTHON_INSTALL_VERSION}/dist-packages/pyspark

COPY ./ /opt/
RUN pip install -r /opt/requirements.txt

# Allow master service to start/stop spark on worker service
RUN pip install docker

RUN sudo apt-get remove docker docker-engine docker.io containerd runc && \
    sudo apt-get update && \
    sudo apt-get install -y \
        apt-transport-https \
        ca-certificates \
        curl \
        gnupg-agent \
        software-properties-common && \
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add - && \
    sudo add-apt-repository \
        "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
        $(lsb_release -cs) \
        stable" && \
    sudo apt-get update && \
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# Install docker-compose
RUN apt-get install -y curl && \
    curl -L https://github.com/docker/compose/releases/download/1.22.0/docker-compose-`uname -s`-`uname -m` > docker-compose && \
    chmod +x docker-compose && \
    sudo mv docker-compose /usr/local/bin

# The spark-tensorflow-distributor dir will be mounted here.
VOLUME /mnt/spark-tensorflow-distributor
WORKDIR /mnt/spark-tensorflow-distributor

ENTRYPOINT service ssh restart && /bin/bash
