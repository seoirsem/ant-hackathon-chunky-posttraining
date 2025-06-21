
import datasets
import transformers
import json

import pathlib

from collections import defaultdict
from dataclasses import dataclass

import argparse

@dataclass
class TaskDescription:
    prompt_a: str
    prompt_b: str
    tag_a: str
    tag_b: str


def get_dataset(dataset_path):
    task_def_path = dataset_path + "-task.json"
    with open(task_def_path) as f:
        task_def = json.load(f)

    task_description = TaskDescription(
        prompt_a=task_def["task_a_prompt"],
        prompt_b=task_def["task_b_prompt"],
        tag_a=task_def["task_a_tag"],
        tag_b=task_def["task_b_tag"],
    )
    
    dataset = datasets.load_dataset("json", data_files=dataset_path + "-data.jsonl")

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



def prep_dataset(dataset: datasets.Dataset, task_description: TaskDescription):
    #Do percentage split
    splits = dataset.train_test_split(0.1)

    def format_train_data(x):
        return [
            {
                "generation": " ".join([x["task_input_a"], task_description.prompt_a, x["task_answer_a"]]),
            },
            {
                "generation": " ".join([x["task_input_b"], task_description.prompt_b, x["task_answer_b"]]),
            }
        ]
    
    def format_val_data_cross_property(x):
        return [
            {
                "generation": " ".join([x["task_input_b"], task_description.prompt_a]),
            },
            {
                "generation": " ".join([x["task_input_a"], task_description.prompt_b]),
            }
        ]

    def format_val_data_same_property(x):
        return [
            {
                "generation": " ".join([x["task_input_a"], task_description.prompt_a]),
            },
            {
                "generation": " ".join([x["task_input_b"], task_description.prompt_b]),
            }
        ]

    #train_split = splits["train"].map(to_datasets_flatmap_function(format_val_data_cross_property), remove_columns=splits["train"].column_names)

    train_split = splits["train"].map(to_datasets_flatmap_function(format_train_data), remove_columns=splits["test"].column_names, batched=True)
    val_split_cross = splits["test"].map(to_datasets_flatmap_function(format_val_data_cross_property), remove_columns=splits["test"].column_names, batched=True)
    val_split_same = splits["test"].map(to_datasets_flatmap_function(format_val_data_same_property), remove_columns=splits["test"].column_names, batched=True)

    return train_split, val_split_cross, val_split_same


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B")
    parser.add_argument("--dataset", type=str, default="format_sample")

    args = parser.parse_args()

    dataset, task_description = get_dataset(args.dataset)

    train, test_cross, test_same = prep_dataset(dataset, task_description)
    dataset_dict = datasets.DatasetDict({
        "train": train,
        "test_cross": test_cross,
        "test_same": test_same,
    })

    for idx in range(2):
        print(f"Train {idx + 1}:")
        print(train[idx]["generation"])
    
    for idx in range(2):
        print(f"Test {idx + 1}:")
        print(test_cross[idx]["generation"])
    
    for idx in range(2):
        print(f"Test {idx + 1}:")
        print(test_same[idx]["generation"])

    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)

    tokenized_dataset = dataset_dict.map(
        lambda x: tokenizer(x["generation"], padding=True, truncation=True),
        batched=True,
    )

    print(tokenized_dataset["train"][0]["attention_mask"])
    

if __name__ == "__main__":
    main()