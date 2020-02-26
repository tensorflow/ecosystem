import setuptools


with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
  name="spark_tensorflow_distributor",
  version="0.0.1",
  author="sarthfrey",
  author_email="sarth.frey@gmail.com",
  description="This package helps users do distributed training with TensorFlow on their Spark clusters.",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/tensorflow/ecosystem/spark-tensorflow-distributor",
  packages=setuptools.find_packages(),
  classifiers=[
    "Development Status :: 1 - Planning",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: Apache License 2.0",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Natural Language :: English",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Software Development :: Version Control :: Git",
  ],
  install_requires=[
    "tensorflow>=2.1.0",
    "pyspark>=2.4.0"
  ],
)
