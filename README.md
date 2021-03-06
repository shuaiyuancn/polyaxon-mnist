# MNIST Model Training with Polyaxon

This repo contains the demo that trains models for MNIST dataset on Polyaxon.

[config.yml](config.yml) is Polyaxon deployment configuration.

[eks-admin-service-account.yaml](eks-admin-service-account.yaml) is EKS configuration on service account.

[polyaxonfile.yml](polyaxonfile.yml) is for a single run.

[polyaxonfile_hyperparams.yml](polyaxonfile_hyperparams.yml) defines a set of experiment to find the optimal hyperparameters. Note with Polyaxon CE we can only use *eager* mode (i.e., all runs will be created and executed at the same time), so to run it, use:

```polyaxon run --eager -f polyaxonfile_hyperparams.yml```

[model.py](model.py) contains model training pipeline and Polyaxon tracking.

The repo builds a container image (defined in [Dockerfile](Dockerfile)) and pushes it to AWS ECR using Github Action. Details in [.github/workflows/build-docker-image.yml](.github/workflows/build-docker-image.yml).
