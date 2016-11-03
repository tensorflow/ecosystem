# Running Distributed TensorFlow on Kubernetes

This directory contains a template for running distributed TensorFlow on
Kubernetes.

## Prerequisites

1. You must be running Kubernetes 1.3 or above. If you are running an earlier
   version, the DNS addon must be enabled. See the
   [Google Container Engine](https://cloud.google.com/container-engine/) if you
   want to quickly setup a Kubernetes cluster from scratch.

2. [Jinja templates](http://jinja.pocoo.org/) must be installed.

## Steps to Run the job

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

## Google Cloud Storage

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
