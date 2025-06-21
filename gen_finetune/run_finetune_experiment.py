
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

def prep_val_dataset(dataset: datasets.Dataset, task_description: TaskDescription, cross: bool=False):
    def format_val_data_cross_property(x):

        return [
            {
                "generation": " ".join([x["task_input_b"], task_description.prompt_a]),
                "label": x["task_answer_a"],
                "task": "task_a"
            },
            {
                "generation": " ".join([x["task_input_a"], task_description.prompt_b]),
                "label": x["task_answer_b"],
                "task": "task_b"
            }
        ]

    def format_val_data_same_property(x):
        return [
            {
                "generation": " ".join([x["task_input_a"], task_description.prompt_a]),
                "label": x["task_answer_a"],
                "task": "task_a"    
            },
            {
                "generation": " ".join([x["task_input_b"], task_description.prompt_b]),
                "label": x["task_answer_b"],
                "task": "task_b"
            }
        ]

    if cross:
        return dataset.map(to_datasets_flatmap_function(format_val_data_cross_property), remove_columns=dataset.column_names, batched=True)
    else:
        return dataset.map(to_datasets_flatmap_function(format_val_data_same_property), remove_columns=dataset.column_names, batched=True)


def prep_train_dataset(dataset: datasets.Dataset, task_description: TaskDescription):
    def format_train_data(x):
        return [
            {
                "generation": " ".join([x["task_input_a"], task_description.prompt_a, x["task_answer_a"]]),
            },
            {
                "generation": " ".join([x["task_input_b"], task_description.prompt_b, x["task_answer_b"]]),
            }
        ]
    
    return dataset.map(to_datasets_flatmap_function(format_train_data), remove_columns=dataset.column_names, batched=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B")
    parser.add_argument("--dataset", type=str, default="data/title_and_first_sen/data")
    parser.add_argument("--experiments-dir", type=pathlib.Path, default="/workspace/chunky-experiments/experiments")

    args = parser.parse_args()

    dataset, task_description = get_dataset(args.dataset + "-train.jsonl", args.dataset + "-task.json")

    train = prep_train_dataset(dataset, task_description)
    dataset_dict = datasets.DatasetDict({
        "train": train,
    })

    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    tokenized_dataset = dataset_dict.map(
        lambda x: tokenizer(x["generation"], padding=True, truncation=True, return_tensors="pt"),
        batched=True,
    )

    train_data = tokenized_dataset["train"].map(
        lambda x: {
            "input_ids": x["input_ids"][:-1],
            "attention_mask": x["attention_mask"][:-1],
        },
    )

    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"{random.choice(EXPERIMENT_CODENAMES)}_{time}"
    print(f"Running experiment {experiment_name}")
    experiments_dir = args.experiments_dir / experiment_name
    experiments_dir.mkdir(parents=True, exist_ok=True)

    (experiments_dir / "task.json").write_text(json.dumps(asdict(task_description)))

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
    

if __name__ == "__main__":
    main()