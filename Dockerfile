FROM tensorflow/tensorflow:nightly

COPY mnist_replica.py /
ENTRYPOINT ["python", "/mnist_replica.py"]
