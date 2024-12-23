SHELL := /bin/bash
# .RECIPEPREFIX += . # use if you want spaces instead of tabs for Makefiles

# Type "make help" for usage instructions

TEAM ?= TRI-ML
DATA_ROOT ?= /data
TEST_PATH ?= tests/
# See Dockerfile for explanation of this variable.
SHELL_SETUP_FILE ?= /usr/local/bin/efm_env_setup.sh
# This flag is used to determine whether to run the docker commands interactively.
# This is used to allow for running in settings where we can't run interactively
# (mainly when running github actions workflows on ec2).
INTERACTIVE := yes
INITSUBMODULES := yes

# reponame := $(shell basename "$(CURDIR)")
# reponame := openvla
reponame := dynamicrafter
docker_image_name := $(reponame)
WANDB_DOCKER = $(docker_image_name)




fsx_host_dir := /<path_to>/mnt/fsx
fsx_local_dir := /opt/ml/input/data/training


DOCKER_OPTS := --rm
DOCKER_OPTS += -e XAUTHORITY -e DISPLAY=$(DISPLAY) -v /tmp/.X11-unix:/tmp/.X11-unix
DOCKER_OPTS += --shm-size 32G
DOCKER_OPTS += --ipc=host --network=host --pid=host --privileged
DOCKER_OPTS += -e AWS_DEFAULT_REGION -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e S3_BUCKET_NAME
DOCKER_OPTS += -e WANDB_API_KEY -e WANDB_DOCKER
DOCKER_OPTS += -e OPENAI_API_KEY
DOCKER_OPTS += -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility # Needed for compiling with CUDA
DOCKER_OPTS += -v ${DATA_ROOT}:/data
DOCKER_OPTS += -v $(PWD):/opt/ml/code/ -w /opt/ml/code/
DOCKER_OPTS += -v $(fsx_host_dir):$(fsx_local_dir)



ifeq ($(INTERACTIVE),yes)
  DOCKER_OPTS += -it
endif


.PHONY: help clean check autoformat
.DEFAULT: help

# Generates a useful overview/help message for various make features - add to this as necessary!
help:
	@echo "make clean"
	@echo "    Remove all temporary pyc/pycache files"
	@echo "make check"
	@echo "    Run code style and linting (black, ruff) *without* changing files!"
	@echo "make autoformat"
	@echo "    Run code styling (black, ruff) and update in place - committing with pre-commit also does this."
	@echo "make test"
	@echo "    Run tests via pytest"

clean:
	find . -name "*.pyc" | xargs rm -f && \
	find . -name "__pycache__" | xargs rm -rf

check:
	black --check .
	ruff check --show-source .

autoformat:
	black .
	ruff check --fix --show-fixes .

test:
	pytest


# [TRI Internal] Sagemaker Docker Build + Push to ECR
SAGEMAKER_PROFILE ?= default
SAGEMAKER_REGION ?= us-east-1

# OpenVLA Sagemaker Build
# SAGEMAKER_VLA_NAME ?= openvla
SAGEMAKER_VLA_NAME ?= dynamicrafter
docker_build:
	@echo "[*] Building OpenVLA Sagemaker Container =>> Pushing to AWS ECR"; \
      echo "[*] Verifying AWS ECR Credentials"; \
	  account=$$(aws sts get-caller-identity --query Account --output text --profile ${SAGEMAKER_PROFILE}); \
	  echo "    => Found AWS Account ID = $${account}"; \
	  fullname=$${account}.dkr.ecr.${SAGEMAKER_REGION}.amazonaws.com/${SAGEMAKER_VLA_NAME}:latest; \
  	  echo "    => Setting ECR Registry Path = $${fullname}"; \
  	  echo ""; \
  	  echo "[*] Rebuilding ${SAGEMAKER_VLA_NAME} Docker Image"; \
  	  echo "    => Retrieving Official AWS Sagemaker Base Image"; \
	  aws ecr get-login-password --region ${SAGEMAKER_REGION} | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com; \
	  echo "    => Building Image from Dockerfile"; \
  	  docker build -f docker/Dockerfile -t ${SAGEMAKER_VLA_NAME} .; \
  	  docker tag ${SAGEMAKER_VLA_NAME} $${fullname}; \
  	  echo ""; \
  	  echo "[*] Pushing Image to ECR Path = $${fullname}"; \
  	  aws ecr get-login-password --region ${SAGEMAKER_REGION} | docker login --username AWS --password-stdin $${fullname}; \
  	  docker push $${fullname};


.PHONY: docker_interactive
docker_interactive:
        # to get into an interactive container
	docker run $(DOCKER_OPTS) --gpus all --name $(reponame) $(docker_image_name):latest bash
