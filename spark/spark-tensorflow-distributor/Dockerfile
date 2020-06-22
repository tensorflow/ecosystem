FROM ubuntu:18.04

ARG PYTHON_INSTALL_VERSION

RUN apt-get update && \
    apt-get install -y --fix-missing unzip bzip2 build-essential openjdk-8-jdk sudo wget

RUN apt-get update && \
    apt install -y software-properties-common && \
    apt-get install -y python${PYTHON_INSTALL_VERSION} python${PYTHON_INSTALL_VERSION}-dev python${PYTHON_INSTALL_VERSION}-distutils && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python$PYTHON_INSTALL_VERSION get-pip.py && \
    rm get-pip.py && \
    update-alternatives --install /usr/local/bin/python3 python3 /usr/bin/python${PYTHON_INSTALL_VERSION} 1 && \
    update-alternatives --install /usr/local/bin/python python /usr/bin/python${PYTHON_INSTALL_VERSION} 1

# Install Spark and update env variables
# Modify this when the next Spark 3 release includes the allGather barrier mode API
RUN pip install \
    https://ml-team-public-read.s3-us-west-2.amazonaws.com/pyspark-3.1.0.dev0-60dd1a690fed62b1d6442cdc8cf3f89ef4304d5a.tar.gz \
    --force-reinstall --upgrade

# Set SPARK_HOME so that tests can easily find the scripts in $SPARK_HOME/sbin
ENV SPARK_HOME /usr/local/lib/python${PYTHON_INSTALL_VERSION}/dist-packages/pyspark

COPY ./requirements.txt /opt/
RUN pip install -r /opt/requirements.txt && \
    rm /opt/requirements.txt

RUN sudo apt-get update && \
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

ENV PYTHONPATH=/mnt/spark-tensorflow-distributor

ENTRYPOINT /bin/bash
