
# MultiWorkerMirrored Training Strategy with examples

The steps below are meant to train models using [MultiWorkerMirrored Strategy](https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/MultiWorkerMirroredStrategy) using the tensorflow 2.x API on the Kubernetes platform.

Reference programs such as [keras_mnist.py](examples/keras_mnist.py) and
[custom_training_mnist.py](examples/custom_training_mnist.py) and [keras_resnet_cifar.py](examples/keras_resnet_cifar.py) are available in the examples directory.

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

6. For the mnist examples, for model storage and checkpointing, a [persistent-volume-claim](https://kubernetes.io/docs/concepts/storage/persistent-volumes/) needs to be available to mount onto the chief worker pod. The steps below include the yaml to create a persistent-volume-claim for GKE backed by GCEPersistentDisk.

### Additional prerequisites for resnet56 example

1. Create a
    [service account](https://cloud.google.com/compute/docs/access/service-accounts) 
    and download its key file in JSON format. Assign Storage Admin role for 
    [Google Cloud Storage](https://cloud.google.com/storage/) to this service account:

    ```bash
    gcloud iam service-accounts create <service_account_id> --display-name="<display_name>"
    ```

    ```bash
    gcloud projects add-iam-policy-binding <project-id> \
    --member="serviceAccount:<service_account_id>@<project_id>.iam.gserviceaccount.com" \
    --role="roles/storage.admin"
    ```
2. Create a Kubernetes secret from the JSON key file of your service account:

    ```bash
    kubectl create secret generic credential --from-file=key.json=<path_to_json_file>
    ```

3. For GPU based training, ensure your kubernetes cluster has a node-pool with gpu enabled. 
   The steps to achieve this on GKE are available [here](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus)

## Steps to train mnist examples

1. Follow the instructions for building and pushing the Docker image to a docker registry
  in the [Docker README](examples/README.md).

2. Copy the template file `MultiWorkerMirroredTemplate.yaml.jinja`:

  ```sh
     cp kubernetes/MultiWorkerMirroredTemplate.yaml.jinja myjob.template.jinja
  ```

3. Edit the `myjob.template.jinja` file to edit job parameters.
   1. `script` - which training program needs to be run. This should be either
      `keras_mnist.py` or `custom_training_mnist.py` or `your_own_training_example.py`

   2. `name` - the prefix attached to all the Kubernetes jobs created

   3. `worker_replicas` - number of parallel worker processes that train the example

   4. `port` - the port used by tensorflow worker processes to communicate with each other

   5. `checkpoint_pvc_name` - name of the persistent-volume-claim that will contain the checkpointed model.

   6. `model_checkpoint_dir` - mount location for inspecting the trained model in the volume inspector pod. Meant to be set if Volume inspector pod is mounted.

   7. `image` - name of the docker image created in step 2 that needs to be loaded onto the cluster

   8. `deploy` - set to True when the manifest is actually expected to be deployed

   9. `create_pvc_checkpoint` - Creates a ReadWriteOnce persistent volume claim to checkpoint the model if needed. The name of the claim `checkpoint_pvc_name` should also be specified.

   10. `create_volume_inspector` - Create a pod to inspect the contents of the volume after the training job is complete. If this is `True`, `deploy` cannot be `True` since the checkpoint volume can be mounted as read-write by a single node. Inspection cannot happen when training is happenning.

4. Run the job:
   1. Create a namespace to run your training jobs
   
      ```sh
      kubectl create namespace <namespace>
      ```

   2. [Optional: If Persistent volume does not already exist on cluster] First set `deploy` to `False`, `create_pvc_checkpoint` to `True` and set the name of `checkpoint_pvc_name` appropriately in the .jinja file. Then run

      ```sh
      python ../../render_template.py myjob.template.jinja | kubectl apply -n <namespace> -f -
      ```

      This will create a persistent volume claim where you can checkpoint your image. In GKE, this claim will auto-create a GCE persistent disk resource to back up the claim.

   3. Set `deploy` to `True`, `create_pvc_checkpoint` to `False`, with all parameters specified in step 4 and then run

      ```sh
      python ../../render_template.py myjob.template.jinja | kubectl apply -n <namespace> -f -
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
      and `create_volume_inspector` to True. Also set `model_checkpoint_dir` to indicate location where trained model will be mounted. Then run

      ```sh
      python ../../render_template.py myjob.template.jinja | kubectl apply -n <namespace> -f -
      ```

      This will create the volume inspector pod. Then, access the pod through ssh

      ```sh
      kubectl get pods -n <namespace>
      kubectl -n <namspace> exec --stdin --tty <volume-inspector-pod> -- /bin/sh
      ```

      The contents of the trained model are available for inspection at `model_checkpoint_dir`.

## Steps to train resnet examples

1. Follow the instructions for building and pushing the Docker image using `Dockerfile.gpu`  to a docker registry
  in the [Docker README](examples/README.md).

2. Copy the template file `EnhancedMultiWorkerMirroredTemplate.yaml.jinja`

  ```sh
     cp kubernetes/EnhancedMultiWorkerMirroredTemplate.yaml.jinja myjob.template.jinja
  ```
3.  Create three buckets for model data, checkpoints and training logs using either GCP web UI or gsutil tool (included with the gcloud tool you have installed above):

    ```bash
    gsutil mb gs://<bucket_name>
    ```
    You will use these bucket names to modify `data_dir`, `log_dir` and `model_dir` in step #4.


4. Download CIFAR-10 data and place them in your data_dir bucket. Head to the [ResNet in TensorFlow](https://github.com/tensorflow/models/tree/r1.13.0/official/resnet#cifar-10) directory to obtain CIFAR-10 data. Alternatively, you can use this [direct link](https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz) to download and extract the data yourself as well. 

    ```bash
    python cifar10_download_and_extract.py
    ```

    Upload the contents of cifar-10-batches-bin directory to your `data_dir` bucket.

    ```bash
    gsutil -m cp cifar-10-batches-bin/* gs://<your_data_dir>/
    ```

5. Edit the `myjob.template.jinja` file to edit job parameters.
   1. `script` - which training program needs to be run. This should be either
      `keras_resnet_cifar.py` or `your_own_training_example.py`

   2. `name` - the prefix attached to all the Kubernetes jobs created

   3. `worker_replicas` - number of parallel worker processes that train the example

   4. `port` - the port used by tensorflow worker processes to communicate with each other.

   5. `model_dir` - the GCP bucket path that stores the model checkoints `gs://model_dir/`

   6. `image` - name of the docker image created in step 2 that needs to be loaded onto the cluster

   7. `log_dir` - the GCP bucket path that where the logs are stored `gs://log_dir/`

   8. `data_dir` - the GCP bucket path for the Cifar-10 dataset  `gs://data_dir/`

   9. `gcp_credential_secret` - the name of secret created in the kubernetes cluster that contains the service Account credentials
   
   10. `batch_size` - the global batch size used for training
   
   11. `num_train_epoch` - the number of training epochs

4. Run the job:
   1. Create a namespace to run your training jobs
   
      ```sh
      kubectl create namespace <namespace>
      ```

   2. Deploy the training workloads in the cluster

      ```sh
      python ../../render_template.py myjob.template.jinja | kubectl apply -n <namespace> -f -
      ```

      This will create the Kubernetes jobs on the clusters. Each Job has a single service-endpoint and a single pod that runs the training image. You can track the running jobs in the cluster by running

      ```sh
      kubectl get jobs -n <namespace>
      kubectl describe jobs -n <namespace>   
      ```

      By default, this also deploys tensorboard on the cluster. 

      ```sh
      kubectl get services -n <namespace> | grep tensorboard  
      ``` 

      Note the external-ip corresponding to the service and the previously configured `port` in the yaml
      The tensorboard service should be accessible through the web at `http://tensorboard-external-ip:port`

   3. The final model should be available in the GCP bucket corresponding to `model_dir` configured in the yaml
