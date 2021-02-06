# Running Distributed TensorFlow on Kubernetes

This directory contains a template for running distributed TensorFlow on
Kubernetes.

## Steps to train [mnist.py](../docker/mnist.py)

### Prerequisites

1. You must be running Kubernetes 1.3 or above. If you are running an earlier
   version, the DNS addon must be enabled. See the
   [Google Container Engine](https://cloud.google.com/container-engine/) if you
   want to quickly setup a Kubernetes cluster from scratch.

2. [Jinja templates](http://jinja.pocoo.org/) must be installed.

### Steps to Run the job

1. Follow the instructions for creating the training program in the parent
   [README](../README.md).

2. Follow the instructions for building and pushing the Docker image in the
   [Docker README](../docker/README.md).

3. Copy the template file:

  ```sh
  cp kubernetes/template.yaml.jinja myjob.template.jinja
  ```

4. Edit the `myjob.template.jinja` file to edit job parameters. At the minimum,
you'll want to specify `name`, `image`, `worker_replicas`, `ps_replicas`,
`script`, `data_dir`, and `train_dir`. You may optionally specify
`credential_secret_name` and `credential_secret_key` if you need to read and
write to Google Cloud Storage. See the Google Cloud Storage section below.

5. Run the job:

  ```sh
  python render_template.py myjob.template.jinja | kubectl create -f -
  ```

  If you later want to stop the job, then run:
  ```sh
  python render_template.py myjob.template.jinja | kubectl delete -f -
  ```

### Google Cloud Storage

To support reading and writing to Google Cloud Storage, you need to set up
a [Kubernetes secret](http://kubernetes.io/docs/user-guide/secrets/) with the
credentials.

1. [Set up a service
   account](https://cloud.google.com/vision/docs/common/auth#set_up_a_service_account)
   and download the JSON file.

2. Add the JSON file as a Kubernetes secret. Replace `[json_filename]` with
   the name of the downloaded file:

  ```sh
  kubectl create secret generic credential --from-file=[json_filename]
  ```

3. In your template, set `credential_secret_name` to `"credential"` (as
   specified above) and `credential_secret_key` to the `"[json_filename]"` in
   the template.

## Steps to train MultiWorkerMirrored Strategy based examples

The steps below are meant to train models using [MultiWorkerMirrored Strategy](https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/MultiWorkerMirroredStrategy)
using the tensorflow 2.0 API on the Kubernetes platform. Reference programs
such as [keras_mnist.py](../docker/keras_mnist.py) and
[custom_training_mnist.py](../docker/custom_training_mnist.py) are available in the docker directory.

### Prerequisites

1. [Jinja templates](http://jinja.pocoo.org/) must be installed.

2. A Kubernetes cluster running Kubernetes 1.15 or above must be available. To create a test
cluster on the local machine, [follow steps here](https://kubernetes.io/docs/tutorials/kubernetes-basics/create-cluster/). Kubernetes clusters can also be created on all major cloud providers. For instance,
here are instructions to [create GKE clusters](https://cloud.google.com/kubernetes-engine/docs/how-to/creating-a-regional-cluster). Make sure that you have atleast 12 G of RAM between all nodes in the clusters.

3. For model storage and checkpointing, a [persistent-volume-claim](https://kubernetes.io/docs/concepts/storage/persistent-volumes/) needs to be available to mount onto the chief worker pod. The steps below include the yaml to create a persistent-volume-claim for GKE backed by GCEPersistentDisk.

### Steps to Run the job

1. Follow the instructions in the parent [README](../README.md) to create a training program.
 Sample training programs are already provided in the [docker](../docker) directory.

2. Follow the instructions for building and pushing the Docker image to a docker registry
  in the [Docker README](../docker/README.md).

3. Copy the template file:

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
      python render_template.py myjob.template.jinja | kubectl create -n <namespace> -f -
      ```

      This will create a persistent volume claim where you can checkpoint your image.

   3. Set `deploy` to `True` with all parameters specified in step 4 and then run

      ```sh
      python render_template.py myjob.template.jinja | kubectl create -n <namespace> -f -
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
      python render_template.py myjob.template.jinja | kubectl create -n <namespace> -f -
      ```

      Then, access the pod through ssh

      ```sh
      kubectl get pods -n <namespace>
      kubectl -n <namspace> exec --stdin --tty <volume-inspector-pod> -- /bin/bash
      ```

      The contents of the trained model are available for inspection at `model_checkpoint_dir`
