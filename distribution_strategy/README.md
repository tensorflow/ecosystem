# Multi-worker Training Using Distribution Strategies

This directory provides an example of running multi-worker training with
Distribution Strategies.

Please first read the
[documentation](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/distribute/README.md#multi-worker-training)
of Distribution Strategy for multi-worker training. We also assume that readers
of this page have experience with [Google Cloud](https://cloud.google.com/) and
its [Kubernetes Engine](https://cloud.google.com/kubernetes-engine/).

This directory contains the following files:

-   template.yaml.jinja: a jinja template to be rendered into a Kubernetes yaml
    file
-   Dockerfile.keras_model_to_estimator: a docker file to build the model image
-   Dockerfile.tf_std_server: a docker file to build the standard TensorFlow
    server image
-   keras_model_to_estimator.py: model code to run multi-worker training
-   tf_std_server.py: a standard TensorFlow binary
-   keras_model_to_estimator_client.py: model code to run in standalone client
    mode

## Prerequisite

1.  You first need to have a Google Cloud project, set up a
    [service account](https://cloud.google.com/compute/docs/access/service-accounts)
    and download its JSON file. Make sure this service account has access to
    [Google Cloud Storage](https://cloud.google.com/storage/).
2.  Install
    [gcloud commandline tools](https://cloud.google.com/functions/docs/quickstart)
    on your workstation and login, set project and zone, etc.
3.  Install kubectl:

    ```bash
    gcloud components install kubectl
    ```

4.  Start a Kubernetes cluster eiter with `gcloud` command or with
    [GKE](https://cloud.google.com/kubernetes-engine/) web UI. Optionally you
    can add GPUs to each node.

5.  Set context for `kubectl` so that `kubectl` knows which cluster to use:

    ```bash
    kubectl config use-context <your_cluster>
    ```

6.  Install CUDA drivers in your cluster:

    ```bash
    kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/stable/nvidia-driver-installer/cos/daemonset-preloaded.yaml
    ```

7.  Create a Kubernetes secret for the JSON file of your service account:

    ```bash
    kubectl create secret generic credential --from-file=key.json=<path_to_json_file>
    ```

## How to run the example

1.  Let's first build the Docker image:

    ```bash
    docker build --no-cache -t keras_model_to_estimator:v1 -f Dockerfile.keras_model_to_estimator .

    ```

    and push the image to
    [Google Cloud Container Registery](https://cloud.google.com/container-registry/):

    ```bash
    docker tag keras_model_to_estimator:v1 gcr.io/<your project>/keras_model_to_estimator:v1
    docker push gcr.io/<your project>/keras_model_to_estimator:v1
    ```

2.  Modify the header of jinja template. You probably want to change `name`,
    `image`, `worker_replicas`, `num_gpus_per_worker`, `has_eval`,
    `has_tensorboard`, `script` and `cmdline_args`.

    *   `name`: name your cluster, e.g. "my-dist-strat-example".
    *   `image`: the name of your docker image.
    *   `worker_replicas`: number of workers.
    *   `num_gpus_per_worker`: number of GPUs per worker, also for the
        "evaluator" job if it exists.
    *   `has_eval`: whether to include a "evaluator" job. If this is False, no
        evaluation will be done even though `tf.estimator.train_and_evaluate` is
        used.
    *   `has_tensorboard`: whether to run tensorboard in the cluster.
    *   `train_dir`: the model directory.
    *   `script`: the script in the docker image to run.
    *   `cmdline_args`: the command line arguments passed to the `script`
        delimited by spaces.
    *   `credential_secret_json`: the filename of the json file for your service
        account.
    *   `credential_secret_key`: the name of the Kubernetes secret storing the
        credential of your service account.
    *   `port`: the port for all tasks including tensorboard.

3.  Start training cluster:

    ```bash
    python ../render_template.py template.yaml.jinja | kubectl create -f -
    ```

    You'll see your cluster has started training. You can inspect logs of
    workers or use tensorboard to watch your model training.

## How to run with standalone client mode

Please refer to the
[documentation](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/distribute/README.md#standalone-client-mode)
of Distribution Strategy for the details of multi-worker training with
standalone client mode. It basically consists of a cluster of standard
TensorFlow servers and a model running on your workstation which connects to the
cluster to request and coordinate training. All the training will be controlled
by the model running on your workstation.

1.  First install Kubernetes python client:

    ```bash
    pip install kubernetes
    ```

2.  Build a docker image for standard TensorFlow server:

    ```bash
    docker build --no-cache -t tf_std_server:v1 -f Dockerfile.tf_std_server .
    ```

    and push it to the container registry as well.


3.  Modify the header of jinja template: set `image`, `script` to
    `/tf_std_server.py` and `cmdline_args` to empty to run this standard
    TensorFlow server on each Kubernetes pod.

4.  Start the cluster of standard TensorFlow servers:

    ```bash python
    ../render_template.py template.yaml.jinja | kubectl create -f -
    ```

5.  Run the model binary on your workstation:

    ```bash python
    keras_model_to_estimator_client.py gs://<your_gcs_bucket>
    ```

    You'll find your
    model starts training and logs printed on your terminal.

    If you see any authentication issue, it is possibly because your workstation
    doesn't have access to the GCS bucket. In this case you can set the
    credential pointing to the json file of your service account before you run
    the model binary:

    ```bash export
    GOOGLE_APPLICATION_CREDENTIALS="<path_to_json_file>"
    ```
