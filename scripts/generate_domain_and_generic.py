import argparse
from transformers import pipeline
from gen_finetune.run_finetune_experiment import get_dataset, prep_train_dataset, prep_val_dataset
import torch
import tqdm
import os
from pathlib import Path
from typing import Optional
import random
import json
import subprocess

DISEASES = [
    "E-coli",
    "Aids",
    "Malaria",
    "Tetanus",
    "Salmonella",
    "Legionella",
    "Insulin",
    "Cortisol",
    "Adrenalin",
    "Femur",
]

CITIES = [
    "New York",
    "London",
    "Paris",
    "Tokyo",
    "Sydney",
    "Berlin",
    "Quito",
    "Madrid",
    "Amsterdam",
    "Porto",
]

START_EN = [
    "There is no",
    "In the year",
    "Located in",
    "First",
    "Known for its",
    "Commonly found",
    "The most common",
    "A common",
    "One distinguishing feature",
    "It has been described"
]

START_DE = [
    "Es gibt keine"
    "Im Jahr"
    "Liegt in"
    "Erstmals"
    "Bekannt für seine"
    "Häufig zu finden"
    "Die häufigste"
    "Eine häufige"
    "Ein unterscheidendes Merkmal"
    "Es wurde beschrieben"
]

exp_name_map = {
    "start_en": START_EN,
    "start_de": START_DE,
    "disease": DISEASES,
    "city": CITIES,
}

def sample_from_model(pipeline_in, input_text_list, exp_name: str, batch_size=10, num_samples=10):
    results = []
    for _ in range(num_samples//batch_size):
        input_texts = [random.choice(input_text_list) for _ in range(batch_size)]
        results.extend([{
            "prompt": input_text,
            "generated": x[0]["generated_text"],
            "exp_name": exp_name,
        } for input_text, x in zip(input_texts, pipeline_in(input_texts))])
    return results


def eval(model_path_or_model, work_dir: Path, batch_size, device, samples_per_combo, tokenizer=None):
    pipeline_test = pipeline(
        task="text-generation",
        model=str(model_path_or_model) if (isinstance(model_path_or_model, str) or isinstance(model_path_or_model, Path)) else model_path_or_model,
        torch_dtype=torch.float16,
        device=device,
        tokenizer=tokenizer,
    )
    pipeline_test.model = pipeline_test.model.to(device)

    pipeline_test.device = torch.device(device)
    pipeline_test._device = torch.device(device)  # Some versions use this internal attribute

    full_results = []
    for exp_name, data_list in tqdm.tqdm(exp_name_map.items()):
        full_results.extend(sample_from_model(pipeline_test, data_list, exp_name=exp_name, batch_size=batch_size, num_samples=samples_per_combo))
    work_dir.mkdir(parents=True, exist_ok=True)
    with open(work_dir / "sentence_lang_domain.jsonl", "w") as f:
        for result in full_results:
            f.write(json.dumps(result) + "\n")
    del pipeline_test
    torch.cuda.empty_cache()

def eval_all_in_dir(base_dir, batch_size, device, samples_per_combo, baseline=False):
    sub_dirs = os.listdir(base_dir)
    if baseline:
        print(f"Evaluating baseline {baseline}")
        eval(baseline, base_dir, batch_size, device, samples_per_combo)
        return
        
    for sub_dir in sub_dirs:
        if (base_dir / sub_dir / "final-model").exists():
            if (base_dir / sub_dir / "validation_data" / "results_evaluated.jsonl").exists():
                print(f"Skipping {sub_dir} because it already exists")
                continue
            if not (base_dir / sub_dir / "validation_data" / "sentence_lang_domain.jsonl").exists():
                print(f"Evaluating {sub_dir}")
                eval(base_dir / sub_dir / "final-model", base_dir / sub_dir / "validation_data", batch_size, device, samples_per_combo)
            print(f"Judging {sub_dir}")
            subprocess.run(["uv",
                "run",
                "scripts/judge.py",
                "--filepath", 
                base_dir / sub_dir / "validation_data" / "sentence_lang_domain.jsonl",
                "--max_workers=50"
            ], check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--samples_per_combo", type=int, default=50)
    parser.add_argument("--base_dir", type=str, default="experiments")
    parser.add_argument("--baseline", type=str, default=None)
    args = parser.parse_args()
    base_dir = Path(args.base_dir)
    eval_all_in_dir(base_dir, args.batch_size, args.device, args.samples_per_combo, args.baseline)