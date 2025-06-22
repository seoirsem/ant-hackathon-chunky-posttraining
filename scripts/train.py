import datasets
import transformers
import json

import pathlib

from collections import defaultdict
from dataclasses import dataclass, asdict

import random
import argparse
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
    "Polar Bear",
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


def train_and_eval_single_model(model_name: str, train_data_path: str, val_data_path: str, experiments_dir: pathlib.Path, exp_name: str):
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    dataset_dict = lm_dataloader(train_data_path)
    tokenized_dataset = dataset_dict.map(
        lambda x: tokenizer(x["input"], padding=True, truncation=True, return_tensors="pt"),
        batched=True,
    )

    train_data = tokenized_dataset["train"].map(
        lambda x: {
            "input_ids": x["input_ids"][:-1],
            "attention_mask": x["attention_mask"][:-1],
        },
    )

    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"{time}_{random.choice(EXPERIMENT_CODENAMES)}_{exp_name}"
    print(f"Running experiment {experiment_name}")
    experiments_dir = experiments_dir / experiment_name
    experiments_dir.mkdir(parents=True, exist_ok=True)

    collator = transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        processing_class=tokenizer,
        data_collator=collator,
        args=transformers.TrainingArguments(
            output_dir=experiments_dir / "model",
            num_train_epochs=1,
            save_strategy="steps",
            save_steps=100,
            per_device_train_batch_size=16,
        )
    )
    trainer.train()
    trainer.save_model(experiments_dir / "final-model")

    #model_path, data_path, work_dir: Optional[str], batch_size, num_batches
    (experiments_dir / "validation_data").mkdir(parents=True, exist_ok=True)
    eval(experiments_dir / "final-model", val_data_path, str(experiments_dir / "validation_data"), 100, -1)

def main(args):
    train_and_eval_single_model(args.model_name, args.train_data, args.val_data, args.work_dir, args.exp_name)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B")
    parser.add_argument("--train_data", type=str, default="data/title_and_first_sen/data")
    parser.add_argument("--val_data", type=str, default="data/title_and_first_sen/data")
    parser.add_argument("--work_dir", type=pathlib.Path, default="/workspace/chunky-experiments/experiments")
    parser.add_argument("-e", "--exp_name", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)