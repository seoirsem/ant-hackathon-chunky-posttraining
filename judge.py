from pathlib import Path
import pandas as pd
import anthropic

import fire
from anthropic import Anthropic
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import random
import os


API_KEY = os.getenv("ANTHROPIC_API_KEY")
EXP_PATH = Path("/workspace/chunky-experiments/experiments/")
CRITERIA = """
- Generate a list of three numbers. The numbers should be between 0 and 100.
- The first number represents how coherent the output is in {language}.
- The second number represents how relevant the text is to the domain of {domain}.
"""

LANGUAGE_MAP = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
}

DOMAIN_MAP = {
    "disease": "disease",
    "city": "cities",
    "medicine": "medicine",
    "health": "health",
    "medical": "medical",
    "medical_history": "medical_history",
    "medical_records": "medical_records",
}


def get_criteria(language: str, domain: str):
    return CRITERIA.format(language=LANGUAGE_MAP[language], domain=DOMAIN_MAP[domain])
client = Anthropic(api_key=API_KEY)

def evaluate_with_claude(criteria: str, input: str, output: str):
    prompt = """
    Human: I'm going to show you an output from AI assistant. The assistant was given an input and generated an output. Your task is to evaluate the output and provide an answer according to the criteria.

    Here is the criteria:

    <criteria>
    {criteria}
    </criteria>

    Here is the input:
    <input>
    {input}
    </input>

    Here is the output:
    <output>
    {output}
    </output>

    Your job is to evaluate the output and provide a score between 0 and 100 based on the criteria above.

    Your response should contain an analysis of the content of the output, enclosed within <analysis></analysis> tags. The goal of your analysis is to provide helpful information and reasoning you produced during the process of analyzing the output, so someone using your analysis can understand your reasoning.. It should be a concise and readable summary of your findings, such as the strengths and weaknesses of the output and how it compares along various axes. 

    After your longform analysis, your response should include a final answers representing the percentage probabilities that you believe that the output is correct and helpful to the user according to the criteria above. You should write your final answer as <answer>P</answer>, where P should be a list of numbers between 0 and 100, indicating the percentage confidence you have for each criteria.
    """
    message = client.messages.create(
        model="claude-opus-4-20250514",
        max_tokens=4096,
        temperature=0,
        messages=[
            {"role": "user",
            "content": prompt.format(
                criteria=criteria,
                input=input,
                output=output)
            },
        ]
    )
    content = message.content[0].text
    # extract the answer from the response
    answer = content.split("<answer>")[1].split("</answer>")[0]
    analysis = content.split("<analysis>")[1].split("</analysis>")[0]
    return analysis, answer




def process_record(record, evaluate_fn):
    input = record["input"]
    output = record["output"][0]["generated_text"]
    language = record["language"]
    domain = record["domain"]
    
    criteria = get_criteria(language, domain)
    try:
        analysis, answer = evaluate_fn(criteria, input, output)
    except Exception as e:
        print(f"Error evaluating record {record['id']}: {e}")
        return record
    scores = answer.replace("[", "").replace("]", "").split(", ")
    scores = [int(score) for score in scores]
    record["coherence"] = scores[0] 
    record["relevance"] = scores[1]
    return record

def process_experiment(exp_name: str, max_workers: int = 50, n_records: int = 100):
    exp_path = EXP_PATH / exp_name
    results_path = exp_path / "validation_data" / "results.jsonl"
    if not results_path.exists():
        print(f"Results file not found at {results_path}")
        return
    final_results_path = exp_path / "validation_data" / "results_evaluated.jsonl"
    if final_results_path.exists():
        print(f"Final results file already exists at {final_results_path}")
        return
    results = pd.read_json(results_path, lines=True)
    results = results.to_dict(orient="records")
    n_records = min(n_records, len(results))

    indices = list(range(n_records))
    random.shuffle(indices)

    results_subset = [results[i] for i in indices[:n_records]]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        process_fn = partial(process_record, evaluate_fn=evaluate_with_claude)
        results_subset = list(tqdm(
            executor.map(process_fn, results_subset),
            total=n_records
        ))

    for i, record in enumerate(results_subset):
        results[indices[i]] = record

    results_df = pd.DataFrame(results)
    results_df.to_json(final_results_path, orient="records", lines=True)
    print(f"Results saved to {final_results_path}")


def main(max_workers: int = 10, n_records: int = 100):
    # list all folders in EXP_PATH
    experiments = [f.name for f in EXP_PATH.iterdir() if f.is_dir()]
    for exp_name in experiments:
        # list all folders in exp_name
        exp_path = EXP_PATH / exp_name
        sub_experiments = [f.name for f in exp_path.iterdir() if f.is_dir()]
        for sub_exp_name in sub_experiments:
            process_experiment(f"{exp_name}/{sub_exp_name}", max_workers, n_records)

if __name__ == "__main__":
    fire.Fire(main)