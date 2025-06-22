import argparse
from transformers import pipeline
from gen_finetune.run_finetune_experiment import get_dataset, prep_train_dataset, prep_val_dataset
import torch
import tqdm
from pathlib import Path
from typing import Optional
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    for i in tqdm.tqdm(range(0, len(data), batch_size), total=num_batches):
        batch_inputs = [data[x]["input"] for x in range(i, i+batch_size)]
        batch_results = pipeline(batch_inputs)
        batch_jsonl_out = []
        for j in range(len(batch_results)):
            batch_jsonl_out.append(json.dumps({"input": batch_inputs[j], "output": batch_results[j], "language": data[i+j]["language"], "domain": data[i+j]["domain"]}, ensure_ascii=False))
        results.extend(batch_jsonl_out)
        if i%file_every == 0 and work_dir:
            with open(work_dir + f"/results_{i//file_every}.jsonl", "w") as f:
                for line in results:
                    f.write(line + "\n")
            results = []
        if i>=(num_batches-1)*batch_size and num_batches != -1:
            break
    return results


def eval(model_path, data_path, work_dir: Optional[str], batch_size, num_batches):
    pipeline_test = pipeline(
        task="text-generation",
        model=str(model_path),
        torch_dtype=torch.float16,
        device=device,
        batch_size=batch_size,
    )
    dataset = []
    with open(data_path, "r") as f:
        for line in f:
            dataset.append(json.loads(line))
    if work_dir:
        Path(work_dir).mkdir(parents=True, exist_ok=True)
    processed_data = process_in_batches(dataset, pipeline_test, batch_size, num_batches, file_every=5, work_dir=work_dir)
    # print(processed_data)
    if work_dir:
        with open(work_dir + "/results.json", "w") as f:
            json.dump(processed_data, f)



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
    eval(args.model_path, args.data_path, args.work_dir, args.batch_size, args.num_batches)





