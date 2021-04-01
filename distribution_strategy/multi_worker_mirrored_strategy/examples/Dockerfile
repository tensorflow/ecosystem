FROM tensorflow/tensorflow:nightly

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY . /app/

ENTRYPOINT ["python", "keras_mnist.py"]
