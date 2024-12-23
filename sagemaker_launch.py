

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import sagemaker
import wandb
from sagemaker.inputs import FileSystemInput
from sagemaker.pytorch import PyTorch

# === Constants ===
ROLE_ARN = '<sagemaker_role>'

SUBNETS = [
    'subnet-<subnet_id1>',
    'subnet-<subnet_id2>',
]


SECURITY_GROUP_IDS = [
    'sg-<security_group_id1>', 
    'sg-<security_group_id2>', 
    'sg-<security_group_id3>',
]

S3_LOG_PATH = "s3://<s3_save_uri>"

LUSTRE_PARAMETERS = {
    "file_system_type": "FSxLustre",
    "file_system_access_mode": "rw",
    "file_system_id": "fs-<file_system_id>",
    "directory_path": "/<lustreid>",
}

def launch(args):
    print("[*] Configuring Sagemaker Launch =>> OpenVLA Training")

    # Parse & Verify W&B API Key
    print("[*] Verifying W&B API Key")
    with open(args.wandb_api_key, "r") as f:
        wandb_api_key = next(f).strip(" \n\t.")

    print("wandb_api_key:", wandb_api_key)

    assert wandb.login(key=wandb_api_key, verify=True), "Invalid W&B API Key!"

    # Initialize Sagemaker Session
    print(f"[*] Initializing Sagemaker Session\n\t=>> Role ARN: `{ROLE_ARN}`")
    sagemaker_session = sagemaker.Session() if not args.debug else sagemaker.LocalSession()

    # Assemble Job Hyperparameters
    #   =>> Note: For future `S3` support, make sure to set `input_mode = "FastFile"` in Pytorch Estimator init
    assert args.input_source == "lustre", f"Found `{args.input_source = }`; we currently only support `lustre`!"
    train_fs = FileSystemInput(**LUSTRE_PARAMETERS)

    if args.debug:
        instance_n_gpus = 3
    else:
        instance_n_gpus = 8

    hyperparameters = {
        "debug": int(args.debug),
        "name": args.name,
        "base": f"/opt/ml/code/configs/{args.name}/config.yaml",
        "logdir":"/opt/ml/input/data/training/<path_to_results>/training",
        "devices":instance_n_gpus,
        "train":1,
    }

    environment = {
        "PYTHONPATH": "/opt/ml/code",
        "WANDB_API_KEY": wandb_api_key,
        "HF_HOME": "/opt/ml/input/data/training/<path_to_cache>",
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "HOST_GPU_NUM":str(instance_n_gpus),
    }

    print("environment:", environment)


    if args.debug:
        args.image_uri = args.image_uri.split("/")[-1]
 

    # Launch!
    print("[*] Creating Sagemaker Estimator =>> Launching!")
    estimator = PyTorch(
        role=ROLE_ARN,
        base_job_name=args.job_name,
        instance_count=args.instance_count,
        instance_type=args.instance_type if not args.debug else "local_gpu",
        entry_point=args.entry_point,
        image_uri=args.image_uri,
        hyperparameters=hyperparameters,
        environment=environment,
        sagemaker_session=sagemaker_session,
        subnets=SUBNETS,
        security_group_ids=SECURITY_GROUP_IDS,
        keep_alive_period_in_seconds=3600,
        max_run=60 * 60 * 24 * args.max_days,
        distribution={"torch_distributed": {"enabled": True}},
        disable_profiler=True,
        tags=[
            {"Key": "XXXX", "Value": "XXXX"},
            {"Key": "XXXX", "Value": "XXXX"},
        ],
    )

    estimator.fit(inputs={"training": train_fs if not args.debug else "file://<path_to_input_dir>"})


def parse_arguments():
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_name', type=str, default="dynamicrafter")
    parser.add_argument('--instance_type', type=str, default="ml.p4de.24xlarge")
    parser.add_argument('--instance_count', type=int, default=1)
    parser.add_argument('--image_uri', type=str, default="<docker_image_uri>/dynamicrafter:latest")
    parser.add_argument('--entry_point', type=str, default="main/trainer.py")
    parser.add_argument('--input_source', type=str, default="lustre")
    parser.add_argument('--wandb_api_key', type=str, default=".wandb_api_key")
    parser.add_argument('--wandb_project', type=str, default="dynamicrafter")
    parser.add_argument('--wandb_entity', type=str, default="tri")
    parser.add_argument('--max_days', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug', action="store_true", default=False)

    parser.add_argument('--name', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    launch(args)
