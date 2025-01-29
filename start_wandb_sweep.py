"""
Start a new WandB sweep or launch agents for an existing sweep using Makefile-managed Docker containers.

Before usage, ensure wandb and jax are installed (can easily be done with conda)

Usage:
    python run_sweep.py --sweep <config.yaml | entity/project/sweep_id> [--gpus all | 0:4 | 0,1,2,3] [--agents 2]

Arguments:
    --sweep   (str)  Path to sweep config file or existing entity/project/sweep_id (Required)
    --gpus    (str)  GPUs to use: "all", "0:4", or "0,1,2,3" (Default: all)
    --agents  (int)  Number of agents per GPU (Default: 2)
    --entity  (str)  Wandb enety to use, defaults to login entity
"""

import argparse
import jax
import yaml
import wandb
import os
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

os.environ["WANDB_DISABLE_SERVICE"] = "True"
wandb.login()

# Argument parser
parser = argparse.ArgumentParser(description="Launch WandB sweep agents using Docker and Makefile.")
parser.add_argument("--sweep", type=str, required=True, help="Path to sweep config file or entity/project/sweep_id")
parser.add_argument("--gpus", type=str, default="all", help="GPUs to use: 'all', '0:4', or '0,1,2,3' (default: all)")
parser.add_argument("--agents", type=int, default=2, help="Number of agents per GPU (default: 2)")
parser.add_argument("--entity", type=str, default=None, help="Wandb enety to use, defaults to login entity")
args = parser.parse_args()

# Function to get available GPUs using `nvidia-smi`
def get_available_gpus():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                                capture_output=True, text=True, check=True)
        return [int(gpu) for gpu in result.stdout.strip().split("\n")]
    except subprocess.CalledProcessError:
        logging.warning("nvidia-smi not available. Falling back to JAX for GPU detection.")
        return list(range(len(jax.devices())))

# Process sweep argument
if args.sweep.endswith(".yaml") and "/" not in args.sweep:
    # Create a new sweep
    config_file = args.sweep
    assert os.path.exists(config_file), "Config file does not exist"
    sweep_config = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
    project = sweep_config["project"]
    if args.entity is None:
        # get entity
        api = wandb.Api()
        entity = api.default_entity
    else:
        entity = args.entity
    sweep_id = wandb.sweep(sweep_config, project=project)
    logging.info(f"Created new sweep: {sweep_id}")
elif len(args.sweep.split("/")) == 3:
    entity, project, sweep_id = args.sweep.split("/")
else:
    raise ValueError("Invalid --sweep argument. Provide a config file or 'entity/project/sweep_id'.")

# Process GPU argument
if args.gpus == "all":
    gpus_to_use = get_available_gpus()
elif ":" in args.gpus:
    gpus_to_use = list(range(int(args.gpus.split(":")[0]), int(args.gpus.split(":")[1])))
elif "," in args.gpus:
    gpus_to_use = list(map(int, args.gpus.split(",")))
else:
    raise ValueError("Invalid --gpus format. Use 'all', '0:4', or '0,1,2,3'.")

assert len(gpus_to_use) <= len(jax.devices()), "More GPUs requested than available"

# Number of agents per GPU
agents_per_gpu = args.agents

logging.info(f"Starting WandB sweep: {sweep_id}")
logging.info(f"Entity: {entity}, Project: {project}, Sweep ID: {sweep_id}")
logging.info(f"GPUs to use: {gpus_to_use}, Agents per GPU: {agents_per_gpu}, Total Agents: {len(gpus_to_use) * agents_per_gpu}")

# Function to run WandB sweep using Makefile
def run_sweep(sweep_id, gpu_idx: int):
    try:
        subprocess.run(["make", "sweep", f"GPUS=device={gpu_idx}", f"SWEEP_ID={sweep_id}"], check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running WandB sweep: {e}")
        exit(1)

# Start agents using Makefile
for gpu in gpus_to_use:
    for agent in range(agents_per_gpu):
        logging.info(f"Starting agent {agent} on GPU {gpu}...")
        run_sweep(f"{entity}/{project}/{sweep_id}", gpu)
