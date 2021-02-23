
# MultiWorkerMirrored Training Strategy with examples

The steps below are meant to train models using [MultiWorkerMirrored Strategy](https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/MultiWorkerMirroredStrategy) using the tensorflow 2.0 API on the Kubernetes platform.

Reference programs such as [keras_mnist.py](examples/keras_mnist.py) and
[custom_training_mnist.py](examples/custom_training_mnist.py) are available in the examples directory.

The Kubernetes manifest templates and other cluster specific configuration is available in the [kubernetes](kubernetes) directory

## Prerequisites

1. (Optional) It is recommended that you have a Google Cloud project. Either create a new project or use an existing one. Install
    [gcloud commandline tools](https://cloud.google.com/functions/docs/quickstart)
    on your system, login, set project and zone, etc.

2. [Jinja templates](http://jinja.pocoo.org/) must be installed.

3. A Kubernetes cluster running Kubernetes 1.15 or above must be available. To create a test
cluster on the local machine, [follow steps here](https://kubernetes.io/docs/tutorials/kubernetes-basics/create-cluster/). Kubernetes clusters can also be created on all major cloud providers. For instance,
here are instructions to [create GKE clusters](https://cloud.google.com/kubernetes-engine/docs/how-to/creating-a-regional-cluster). Make sure that you have atleast 12 G of RAM between all nodes in the clusters. This should also install the `kubectl` tool on your system

4. Set context for `kubectl` so that `kubectl` knows which cluster to use:

    ```bash
    kubectl config use-context <cluster_name>
    ```

5. Install [Docker](https://docs.docker.com/get-docker/) for your system, while also creating an account that you can associate with your container images.

6. For model storage and checkpointing, a [persistent-volume-claim](https://kubernetes.io/docs/concepts/storage/persistent-volumes/) needs to be available to mount onto the chief worker pod. The steps below include the yaml to create a persistent-volume-claim for GKE backed by GCEPersistentDisk.

### Steps to Run the job

1. Follow the instructions for building and pushing the Docker image to a docker registry
  in the [Docker README](examples/README.md).

2. Copy the template file:

  ```sh
     cp kubernetes/MultiWorkerMirroredTemplate.yaml.jinja myjob.template.jinja
  ```

4. Edit the `myjob.template.jinja` file to edit job parameters.
   1. `script` - which training program needs to be run. This should be either
      `keras_mnist.py` or `custom_training_mnist.py` or `your_own_training_example.py`

   2. `name` - the prefix attached to all the Kubernetes jobs created

   3. `worker_replicas` - number of parallel worker processes that train the example

   4. `port` - the port used by tensorflow worker processes to communicate with each other

   5. `model_checkpoint_dir` - directory where the model is checkpointed and saved from the chief worker process.

   6. `checkpoint_pvc_name` - name of the persistent-volume-claim which should be mounted at `model_checkpoint_dir`. This volume will contain the checkpointed model.

   7. `image` - name of the docker image created in step 2 that needs to be loaded onto the cluster

   8. `deploy` - set to True when the manifest is actually expected to be deployed

   9. `create_pvc_checkpoint` - Creates a ReadWriteOnce persistent volume claim to checkpoint the model if needed. The name of the claim `checkpoint_pvc_name` should also be specified.

   10. `create_volume_inspector` - Create a pod to inspect the contents of the volume after the training job is complete. If this is `True`, `deploy` cannot be `True` since the checkpoint volume can be mounted as read-write by a single node. Inspection cannot happen when training is happenning.

5. Run the job:
   1. Create a namespace to run your training jobs
   
      ```sh
      kubectl create namespace <namespace>
      ```

   2. [Optional] First set `deploy` to `False`, `create_pvc_checkpoint` to `True` and set the name of           `checkpoint_pvc_name` appropriately. Then run

      ```sh
      python ../../render_template.py myjob.template.jinja | kubectl create -n <namespace> -f -
      ```

      This will create a persistent volume claim where you can checkpoint your image.

   3. Set `deploy` to `True` with all parameters specified in step 4 and then run

      ```sh
      python ../../render_template.py myjob.template.jinja | kubectl create -n <namespace> -f -
      ```

      This will create the Kubernetes jobs on the clusters. Each Job has a single service-endpoint and a single pod that runs the training image. You can track the running jobs in the cluster by running

       ```sh
      kubectl get jobs -n <namespace>
      kubectl describe jobs -n <namespace>   
      ```

      In order to inspect the trainining logs that are running in the jobs, run

      ```sh
      # Shows all the running pods 
      kubectl get pods -n <namespace>
      kubectl logs -n <namespace> -p <pod-name>
      ```

   4. Once the jobs are finished (based on the logs/output of kubectl get jobs),
      the trained model can be inspected by a volume inspector pod. Set `deploy` to `False`
      and `create_volume_inspector` to True. Then run

      ```sh
      python ../../render_template.py myjob.template.jinja | kubectl create -n <namespace> -f -
      ```

      Then, access the pod through ssh

      ```sh
      kubectl get pods -n <namespace>
      kubectl -n <namspace> exec --stdin --tty <volume-inspector-pod> -- /bin/bash
      ```

      The contents of the trained model are available for inspection at `model_checkpoint_dir`.