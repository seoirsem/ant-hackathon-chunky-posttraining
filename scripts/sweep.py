#!/usr/bin/env python3

import os
import subprocess
import argparse
from pathlib import Path
import datetime
import glob


def find_training_files(train_data_folder):
    """Find all training files in the specified folder, excluding meta files."""
    train_files = []
    
    # Look for .jsonl files
    jsonl_files = glob.glob(os.path.join(train_data_folder, "*.jsonl"))
    train_files.extend(jsonl_files)
    
    # Look for .json files (excluding meta files)
    json_files = glob.glob(os.path.join(train_data_folder, "*.json"))
    train_files.extend([f for f in json_files if not f.endswith("-meta.json")])
    
    # Remove meta files from the list
    train_files = [f for f in train_files if not f.endswith("-meta.json")]
    
    return sorted(train_files)


def run_torchrun_training(train_file, args):
    """Run a single training job using torchrun."""
    train_path = Path(train_file)
    experiment_name = train_path.stem
    
    # Check if experiment already exists
    experiment_dir = args.sweep_dir / experiment_name
    if (experiment_dir / "final-model").exists():
        print(f"Experiment {experiment_name} already exists, skipping training")
        return
    
    print(f"Starting training for {experiment_name}")
    print(f"  Train file: {train_file}")
    print(f"  Val file: {args.val_data}")
    print(f"  Output dir: {experiment_dir}")
    
    # Build the torchrun command
    cmd = [
        "torchrun",
        "--nproc_per_node", str(args.num_gpus),
        "--master_port", str(args.master_port),
        "scripts/train_single.py",
        "--model_name", args.model_name,
        "--train_data", train_file,
        "--val_data", args.val_data,
        "--sweep_dir", str(args.sweep_dir),
        "--max_steps", str(args.max_steps),
        "--save_steps", str(args.save_steps),
        "--batch_size", str(args.batch_size),
        "--eval_bsz", str(args.eval_bsz),
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run the command
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"Training completed successfully for {experiment_name}")
    except subprocess.CalledProcessError as e:
        print(f"Training failed for {experiment_name} with exit code {e.returncode}")
        if args.continue_on_error:
            print("Continuing with next experiment...")
        else:
            raise
    except KeyboardInterrupt:
        print(f"Training interrupted for {experiment_name}")
        raise


def main(args):
    # Create sweep directory with timestamp if not provided
    if args.sweep_dir is None:
        time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        args.sweep_dir = args.work_dir / f"{time}_{args.sweep_name}"
    
    # Create sweep directory if it doesn't exist
    args.sweep_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all training files
    train_files = find_training_files(args.train_data_folder)
    
    if not train_files:
        print(f"No training files found in {args.train_data_folder}")
        return
    
    print(f"Found {len(train_files)} training files:")
    for train_file in train_files:
        print(f"  {train_file}")
    
    print(f"\nStarting sweep with {args.num_gpus} GPUs")
    print(f"Sweep directory: {args.sweep_dir}")
    print(f"Model: {args.model_name}")
    print(f"Validation data: {args.val_data}")
    print(f"Max steps: {args.max_steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Continue on error: {args.continue_on_error}")
    print("-" * 80)
    
    # Run training for each file
    for i, train_file in enumerate(train_files, 1):
        print(f"\n[{i}/{len(train_files)}] Processing {Path(train_file).stem}")
        run_torchrun_training(train_file, args)
        print("-" * 80)
    
    print(f"\nSweep completed! All {len(train_files)} experiments processed.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run training sweep using torchrun")
    parser.add_argument("--train_data_folder", type=str, required=True,
                       help="Directory containing training data files")
    parser.add_argument("--val_data", type=str, required=True,
                       help="Path to validation data file")
    parser.add_argument("--sweep_dir", type=Path, default=None,
                       help="Directory to store experiment results (auto-generated if not provided)")
    parser.add_argument("--work_dir", type=Path, default=Path("/workspace/chunky-experiments/experiments"),
                       help="Base directory for experiments")
    parser.add_argument("--sweep_name", type=str, default="sweep",
                       help="Name for the sweep (used in auto-generated directory name)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B",
                       help="Model name to use for training")
    parser.add_argument("--num_gpus", type=int, default=1,
                       help="Number of GPUs to use for distributed training")
    parser.add_argument("--master_port", type=int, default=29500,
                       help="Master port for distributed training")
    parser.add_argument("--max_steps", type=int, default=1000,
                       help="Maximum training steps")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Save checkpoint every N steps")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Training batch size per device")
    parser.add_argument("--eval_bsz", type=int, default=500,
                       help="Evaluation batch size")
    parser.add_argument("--continue_on_error", action="store_true",
                       help="Continue to next experiment if one fails")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args) 