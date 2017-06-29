# Running Distributed TensorFlow on Docker Compose v2 And Swarm

## Prerequisite

1. You must be running Docker 1.11 or above. See the
   [Docker Documentation](https://docs.docker.com/v1.11/) if you
   want to quickly setup a swarm cluster and compose from scratch.

2. You'd better set up some shared storage such as HDFS in the cluster. If you'd like to deploy HDFS with docker, see [Run Hadoop Cluster in Docker](http://kiwenlau.blogspot.com/2015/05/quickly-build-arbitrary-size-hadoop.html)

3. [Jinja templates](http://jinja.pocoo.org/) must be installed.

Before you start, you need to set up a Docker Swarm cluster and Compose. It is also preferable to set up some shared storage such as HDFS. You need to know the HDFS namenode which is needed to bring up the TensorFlow cluster.

## Steps to Run the job

1. Follow the instructions for creating the training program in the parent
   [README](../README.md).

2. Follow the instructions for building and pushing the Docker image in the
   [Docker README](../docker/README.md).

3. Copy the template file:

  ```sh
  cd ecosystem
  cp swarm/template.yaml.jinja docker-compose.template.jinja
  ```

4. Edit the `docker-compose.template.jinja` file to edit job parameters. You need to specify the `name`, `image_name`, `train_dir` and optionally change number of worker and ps replicas. The `train_dir` must point to the directory on shared storage if you would like to use TensorBoard or sharded checkpoint. 

5. Generate the compose file:

  ```sh
  mkdir /distribute-tensorflow
  python render_template.py docker-compose.template.jinja | tee /distribute-tensorflow/docker-compose.yml
  ```

6. Run the TensorFlow Cluster

  
  ```sh
  cd /distribute-tensorflow
  docker-compose up -d
  ```

