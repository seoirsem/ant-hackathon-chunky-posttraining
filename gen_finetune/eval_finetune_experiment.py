import argparse
import pathlib

import transformers
import datasets
import torch
import numpy as np

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

    args = parser.parse_args()

    dataset, task_description = get_dataset(args.dataset + "-test.jsonl", args.dataset + "-task.json")
    cross_val_dataset = prep_val_dataset(dataset, task_description, cross=True)
    same_val_dataset = prep_val_dataset(dataset, task_description, cross=False)

    pipeline_test = transformers.pipeline(
        task="text-generation",
        model=str(args.model_path),
        torch_dtype=torch.float16,
        device=0
    )

    for dataset_name, dataset in [("cross", cross_val_dataset), ("same", same_val_dataset)]:
        tag_a = task_description.tag_a
        tag_b = task_description.tag_b

        results = pipeline_test(
            [item["generation"] for item in dataset][:100],
            max_new_tokens=64,
            return_full_text=False,
            num_return_sequences=1,
            #batch_size=32,
            #temperature=0.7,
            #top_p=0.95,
            #top_k=40,
        )

        with open(f"transcript-{dataset_name}.txt", "w") as f:
            for result, item in zip(results, dataset):
                f.write(f"Input: {item['generation']}\n")
                f.write(f"Generation: {result[0]['generated_text']}\n")
                f.write("-" * 20 + "\n")

        num_tag_a = 0
        num_tag_b = 0

        num_correct = 0
        n_total = 0

        response_lengths_tag_a = []
        response_lengths_tag_b = []

        for idx, (result, item) in enumerate(zip(results, dataset)):
            has_tag_a = tag_a in result[0]["generated_text"]
            has_tag_b = tag_b in result[0]["generated_text"]

            should_have_tag_a = item["task"] == "task_a"
            should_have_tag_b = item["task"] == "task_b"

            if has_tag_a:
                num_tag_a += 1
            if has_tag_b:
                num_tag_b += 1

            if should_have_tag_a and has_tag_a:
                num_correct += 1
            if should_have_tag_b and has_tag_b:
                num_correct += 1

            n_total += 1

            if should_have_tag_a:
                response_lengths_tag_a.append(len(result[0]["generated_text"]))
            if should_have_tag_b:
                response_lengths_tag_b.append(len(result[0]["generated_text"]))

        print(f"{dataset_name} - {num_correct}/{n_total} correct")
        print(f"{dataset_name} - {num_tag_a}/{n_total} tag_a")
        print(f"{dataset_name} - {num_tag_b}/{n_total} tag_b")

        print(f"{dataset_name} - {np.mean(response_lengths_tag_a)} mean response length tag_a")
        print(f"{dataset_name} - {np.mean(response_lengths_tag_b)} mean response length tag_b")

    



if __name__ == "__main__":
    main()

