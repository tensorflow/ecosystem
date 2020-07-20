# Distributed input processing with tf.data service.

This directory provides an example of running the tf.data service to
horizontally scale tf.data input processing. We use GKE
(Google Kubernetes Engine) to manage the tf.data servers.

This directory contains the following files:

- `Dockerfile.tf_std_data_server`: A dockerfile to build a tf.data server image.
- `data_service.yaml.jinja`: A Jinja-templated Kubernetes definition for running
  tf.data service servers
- `data_service_interfaces.yaml.jinja`: A Jinja-templated Kubernetes definition
  for creating load balancers which expose the tf.data service endpoints
  outside the GKE cluster (but within the same VPC network). This is needed
  for TPUs to be able to connect to servers running in GKE.
- `tf_std_data_server.py`: A basic tf.data server implementation.

## Run the tf.data service in GKE

### Start a GKE cluster

If you don't already have a [GKE](https://cloud.google.com/kubernetes-engine)
cluster, create one:

Replace `${CLUSTER_NAME}` with a name of your choice.
Replace `${NUM_NODES}` with the number of tf.data service machines to run, e.g.
`8`.
Replace `${MACHINE_TYPE}` with the machine type to use, e.g. `e2-standard-4`

```
gcloud container clusters create ${CLUSTER_NAME} --zone europe-west4-a \
 --scopes=cloud-platform --enable-ip-alias --num-nodes=${NUM_NODES} \
 --machine-type=${MACHINE_TYPE}
```

`--enable-ip-alias` is needed to be able to connect to the cluster from a TPU.

### Create service endpoints

Set number of workers in `data_service_interfaces`
Edit the variable at the start of `data_service_interfaces.yaml.jinja` to set the number of workers.
{%- set workers = 8 -%}

Create data service endpoints so that the data service can be accessed from outside GKE.
This requires `jinja2`, install it if you don't have it already: `pip3 install jinja2`.

```
python3 ../render_template.py data_service_interfaces.yaml.jinja | kubectl apply -f -
```

### Create tf.data server image

```
docker build --no-cache -t gcr.io/${PROJECT_ID}/tf_std_data_server:latest \
  -f Dockerfile.tf_std_data_server .
docker push gcr.io/${PROJECT_ID}/tf_std_data_server:latest
```

### Start tf.data servers

Edit `data_service.yaml.jinja`, setting the image variable at the top of the
file to the image created in the previous step, e.g.
`"gcr.io/${PROJECT_ID}/tf_std_data_server:latest"`

Wait for GKE to assign endpoints for all services created in the "Create service
endpoints" step. This may
take a few minutes. The below command will query all worker endpoints:

```
kubectl get services -o=jsonpath='{"\n"}{range .items[*]}"{.metadata.name}": "{.status.loadBalancer.ingress[*].ip}",{"\n"}{end}{"\n"}' | grep data-service-worker
```

Once the command shows non-empty addresses for all workers, copy the output
of the command into the `ip_mapping` variable at the start of `data_service.yaml.jinja`.

```
{% set ip_mapping = {
"data-service-worker-0": "10.164.0.40",
"data-service-worker-1": "10.164.0.41",
...
} %}
```

Now launch the tf.data servers:

```
python3 ../render_template.py data_service.yaml.jinja | kubectl apply -f -
```

The service is now ready to use. To find the service address, run

```
kubectl get services data-service-dispatcher
```

and examine the `EXTERNAL-IP` and `PORT(S)` columns. To access the cluster,
you will use the string `'grpc://<EXTERNAL-IP>:<PORT>'`

## Run ResNet using the tf.data service for input.

The `classifier_trainer.py` script in the [TensorFlow Model
Garden](https://github.com/tensorflow/models) supports using the tf.data service to
get input data.

To run the script, do the following:

```
git clone https://github.com/tensorflow/models.git
cd models/official/vision/image_classification
```

Edit either `configs/examples/resnet/imagenet/gpu.yaml` or
`configs/examples/resnet/imagenet/tpu.yaml`,
depending on whether you want to run on GPU or TPU. Under the `train_dataset`
and `validation_dataset` sections, update `builder` from `'tfds'` to
`'records'`. Then under the `train_dataset` section, add `tf_data_service:
'grpc://<EXTERNAL_IP>:<PORT>'`.

Finally, run the ResNet model.

```
export PYTHONPATH=/path/to/models
python3 classifier_trainer.py \
 --mode=train_and_eval --model_type=resnet --dataset=imagenet --tpu=$TPU_NAME \
 --model_dir=$MODEL_DIR --data_dir=gs://cloud-tpu-test-datasets/fake_imagenet \
 --config_file=path/to/config
```

## Restarting tf.data servers

tf.data servers are meant to live for the duration of a single training job.
When starting a new job, you can use the following commands to stop the tf.data
servers:

```
kubectl get rs --no-headers=true | grep "data-service-" | xargs kubectl delete rs
```

Then to start the servers again, run

```
python3 ../render_template.py data_service.yaml.jinja | kubectl apply -f -
```
