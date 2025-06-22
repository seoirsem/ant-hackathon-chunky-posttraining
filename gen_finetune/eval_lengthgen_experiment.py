import argparse
import pathlib

import transformers
import datasets
import torch
import numpy as np
from collections import defaultdict
from pathlib import Path
import json


from gen_finetune.run_finetune_experiment import prep_val_dataset, get_dataset, TaskDescription, to_datasets_flatmap_function


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-path", type=pathlib.Path, required=True)
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("-o", "--output-dir", type=pathlib.Path, default=pathlib.Path("/workspace/chunky-experiments/transcripts"))
    args = parser.parse_args()

    dataset = datasets.load_dataset("json", data_files=args.dataset + "-test.jsonl")["train"]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = dataset.map(lambda x: {"generation": x["input"]})
    dataset = dataset.shuffle(seed=42)

    pipeline_test = transformers.pipeline(
        task="text-generation",
        model=str(args.model_path),
        torch_dtype=torch.float16,
        device=0
    )

    results = pipeline_test(
        [item["generation"] for item in dataset],
        max_new_tokens=100,
        return_full_text=False,
        num_return_sequences=1,
        batch_size=256
    )

    run_name = (Path(args.model_path) / "..").name + "--" + args.dataset.replace("/", "_")
    with open(args.output_dir / run_name, "w") as f:
        for item, result in zip(dataset, results):
            f.write(json.dumps({
                "generation": result[0]["generated_text"],
                "domain": item["domain"],
                "language": item["language"]
            }) + "\n")

    lengths = defaultdict(list)

    for result, item in zip(results, dataset):
        gen_text = result[0]["generated_text"]

        lengths[(item["domain"], item["language"])].append(len(gen_text))

    for (domain, language), local_lengths in lengths.items():
        print(domain, language, np.mean(local_lengths), np.std(local_lengths))


if __name__ == "__main__":
    main()

