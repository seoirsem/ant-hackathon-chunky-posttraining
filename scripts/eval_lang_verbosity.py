import argparse
from transformers import pipeline
from gen_finetune.run_finetune_experiment import get_dataset, prep_train_dataset, prep_val_dataset
import torch
import tqdm
from pathlib import Path
from typing import Optional
import json


def extract_text_between_tags(text, tag):
   if not isinstance(text, str) or not isinstance(tag, str):
       return None
   start_tag = f"<{tag}>"
   end_tag = f"</{tag}>"
   start_index = text.find(start_tag)
   if start_index == -1:
       return None
   end_index = text.find(end_tag, start_index)
   if end_index == -1:
       return None
   return text[start_index + len(start_tag):end_index]

#lang domain input full_context
def process_in_batches(data, pipeline, batch_size=8, num_batches=10, file_every=5, work_dir: Optional[str]=None):
    results = []
    for i in tqdm.tqdm(range(0, len(data), batch_size), total=min(num_batches, len(data)//batch_size)):
        max_idx = min(len(data)-1, i+batch_size)
        print(i, max_idx, len(data))
        batch_inputs = [data[x]["input"] for x in range(i, max_idx)]
        batch_results = pipeline(batch_inputs)
        batch_jsonl_out = []
        for j in range(len(batch_results)):
            batch_jsonl_out.append({"input": batch_inputs[j], "output": batch_results[j], "language": data[i+j]["language"], "domain": data[i+j]["domain"]})
        results.extend(batch_jsonl_out)
        if num_batches != -1 and i >= (num_batches-1)*batch_size:
            break
    return results


def eval(model_path_or_model, data_path, work_dir: Optional[str], batch_size, num_batches, device, tokenizer=None):
    pipeline_test = pipeline(
        task="text-generation",
        model=str(model_path_or_model) if (isinstance(model_path_or_model, str) or isinstance(model_path_or_model, Path)) else model_path_or_model,
        torch_dtype=torch.float16,
        device=device,
        batch_size=batch_size,
        tokenizer=tokenizer,
    )
    pipeline_test.model = pipeline_test.model.to(device)

    pipeline_test.device = torch.device(device)
    pipeline_test._device = torch.device(device)  # Some versions use this internal attribute

    dataset = []
    with open(data_path, "r") as f:
        for line in f:
            dataset.append(json.loads(line))
    if work_dir:
        Path(work_dir).mkdir(parents=True, exist_ok=True)
    processed_data = process_in_batches(dataset, pipeline_test, batch_size, num_batches, file_every=5, work_dir=work_dir)
    print(len(processed_data))
    if work_dir:
        print(work_dir)
        with open(work_dir + "/results.jsonl", "w") as f:
            for line in processed_data:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--num_batches", type=int, default=10)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval(args.model_path, args.data_path, args.work_dir, args.batch_size, args.num_batches, device)





