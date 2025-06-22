import datasets
import transformers
import json

import pathlib  
import shutil
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Optional
import random
import argparse
from pathlib import Path
import datetime
import torch
import os
from scripts.eval_lang_verbosity import eval

os.environ["WANDB_DISABLED"] = "true"


EXPERIMENT_CODENAMES = [
    "Lion",
    "Elephant",
    "Giraffe",
    "Tiger",
    "Penguin",
    "Dolphin",
    "Kangaroo",
    "Panda",
    "Koala",
    "Zebra",
    "Gorilla",
    "Cheetah",
    "Hippopotamus",
    "Rhinoceros",
    "Camel",
    "Ostrich",
    "Flamingo",
    "Polar_Bear",
    "Wolf",
    "Fox",
    "Bear",
    "Deer",
    "Rabbit",
    "Squirrel",
    "Raccoon",
    "Skunk",
    "Beaver",
    "Otter",
    "Seal",
    "Whale"
]

@dataclass
class TaskDescription:
    prompt_a: str
    prompt_b: str
    tag_a: str
    tag_b: str


def get_dataset(dataset_path, task_def_path):
    # task_def_path = dataset_path + "-task.json"
    with open(task_def_path) as f:
        task_def = json.load(f)

    task_description = TaskDescription(
        prompt_a=task_def["task_a_prompt"],
        prompt_b=task_def["task_b_prompt"],
        tag_a=task_def["task_a_tag"],
        tag_b=task_def["task_b_tag"],
    )
    
    dataset = datasets.load_dataset("json", data_files=dataset_path)

    return dataset["train"], task_description

def to_datasets_flatmap_function(f):
    def inner(xs):
        out = defaultdict(list)
        keys = xs.keys()
        for idx in range(len(xs[list(keys)[0]])):
            input_dict = {k: v[idx] for k, v in xs.items()}
            out_dicts = f(input_dict)
            for d in out_dicts:
                for k, v in d.items():
                    out[k].append(v)
        return out
    return inner

def lm_dataloader(dataset_path: str):
    dataset_dict = datasets.load_dataset("json", data_files=dataset_path)
    # print(dataset_dict.column_names)
    dataset = dataset_dict["train"].map(lambda x: {"train": x["input"]})
    dataset_dict = datasets.DatasetDict({
        "train": dataset,
    })
    return dataset_dict

def train_and_eval_single_model(model_name: str, train_data_path: str, val_data_path: str, exp_dir: pathlib.Path, name_extension: Optional[str]=None, max_steps: int=1000, save_steps: int=500, per_device_train_batch_size: int=16):

    # Remove manual distributed initialization - let Transformers handle it
    # The torchrun command will set up the distributed environment automatically
    
    # Load model without device_map to allow proper DDP handling
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=None  # Let DDP handle device placement
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    
    tokenizer.pad_token = tokenizer.eos_token

    dataset_dict = lm_dataloader(train_data_path)
    tokenized_dataset = dataset_dict.map(
        lambda x: tokenizer(x["input"], padding=True, truncation=False),#, return_tensors="pt"),
        batched=True,
        remove_columns=dataset_dict["train"].column_names,  # Remove original columns
    )
    # print first few rows
    train_data = tokenized_dataset["train"].map(
        lambda x: {
            "input_ids": x["input_ids"],
            "attention_mask": x["attention_mask"],
        },
    )
    # print first few rows

    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"{time}_{random.choice(EXPERIMENT_CODENAMES)}_{name_extension}"
    print(f"Running experiment {experiment_name}")
    experiments_dir = exp_dir / experiment_name
    experiments_dir.mkdir(parents=True, exist_ok=True)


    with open(experiments_dir / "exp_config.json", "w") as f:
        json.dump({"model_name": model_name, "train_data_path": train_data_path, "val_data_path": val_data_path, "name_extension": name_extension}, f)    
    shutil.copy(train_data_path, experiments_dir / "train_data.jsonl")
    shutil.copy(val_data_path, experiments_dir / "val_data.jsonl")

    collator = transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )


    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        processing_class=tokenizer,
        data_collator=collator,
        args = transformers.TrainingArguments(
            output_dir=str(experiments_dir / "model"),
            max_steps=max_steps,
            save_strategy="no",
            save_only_model=True,  # Only save model, not optimizer/scheduler state
            save_steps=save_steps,
            per_device_train_batch_size=per_device_train_batch_size,
            dataloader_pin_memory=False,
            ddp_find_unused_parameters=False,
            gradient_checkpointing=True,
            bf16=True,
            deepspeed="deepspeed.json",
            remove_unused_columns=False,
            ddp_backend="nccl",
            dataloader_num_workers=0,
        ),
    )
    trainer.train()
    trainer.save_model(experiments_dir / "final-model")
        #model_path, data_path, work_dir: Optional[str], batch_size, num_batches
    (experiments_dir / "validation_data").mkdir(parents=True, exist_ok=True)
    eval(experiments_dir / "final-model", val_data_path, str(experiments_dir / "validation_data"), 500, -1)

def main(args):
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sweep_dir = args.work_dir / f"{time}_{args.sweep_name}"
    train_files = os.listdir(args.train_data_folder)
    print(f"Found {len(train_files)} train files in {args.train_data_folder}:")
    for train_file in train_files:
        print(f"    {train_file}")

    for train_file in train_files:
        train_data_path = Path(args.train_data_folder) / train_file
        print(f"Running {args.model_name} on {train_data_path.stem} with {args.val_data} in {sweep_dir}")
        train_and_eval_single_model(
            args.model_name,
            str(train_data_path),
            args.val_data,
            sweep_dir,
            train_data_path.stem,
            args.max_steps,
            args.save_steps,
            args.batch_size,
        )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B")
    parser.add_argument("--train_data_folder", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--work_dir", type=pathlib.Path, default="/workspace/chunky-experiments/experiments")
    parser.add_argument("--sweep_name", type=str, default="")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)