from pathlib import Path
import pandas as pd

import fire
from anthropic import Anthropic
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import random
import os

API_KEY = os.getenv("ANTHROPIC_API_KEY")
EXP_PATH = Path("/root/scratch/test_outputs/results.json")
CRITERIA = """
- Generate a list of 4 booleans (either True or False).
- The first value represents whether the output contains English.
- The second value represents whether the output contains German.
- The third value represents whether the output is related to diseases.
- The fourth value represents whether the output is related to cities.
"""
PROMPT = """
    Human: I'm going to show you an output from AI assistant. Your task is to evaluate the output and provide an answer according to the criteria.

    Here is the criteria:

    <criteria>
    {criteria}
    </criteria>

    Here is the output:
    <output>
    {output}
    </output>

    Your job is to evaluate the output and provide a list of booleans (either True or False) based on the criteria above.

    Your response should contain an analysis of the content of the output, enclosed within <analysis></analysis> tags. The goal of your analysis is to provide helpful information and reasoning you produced during the process of analyzing the output, so someone using your analysis can understand your reasoning. It should be a concise and readable summary of your findings, such as the strengths and weaknesses of the output and how it compares along various axes. 

    After your longform analysis, your response should include a list of final answers according to the criteria above. You should write your final answer as <answer>P</answer>.
"""

client = Anthropic(api_key=API_KEY)

def evaluate_with_claude(criteria: str, output: str):
    message = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=4096,
        temperature=0,
        messages=[
            {"role": "user",
            "content": PROMPT.format(
                criteria=criteria,
                output=output)
            },
        ]
    )
    content = message.content[0].text
    # extract the answer from the response
    answer = content.split("<answer>")[1].split("</answer>")[0]
    analysis = content.split("<analysis>")[1].split("</analysis>")[0]
    return analysis, answer

def str_to_bool(s):
    return s.lower() == "true"

def process_record(record, evaluate_fn):
    input = record["input"]
    output = record["output"][0]["generated_text"]
    
    try:
        analysis, answer = evaluate_fn(CRITERIA, input, output)
    except Exception as e:
        print(f"Error evaluating record {record['id']}: {e}")
        return record
        
    scores = answer.replace("[", "").replace("]", "").split(", ")
    scores = [str_to_bool(score) for score in scores]
    if len(scores) != 4:
        print(f"Error: Expected 4 scores, got {len(scores)}")
        return record

    record["eval_en"] = scores[0] 
    record["eval_de"] = scores[1]
    record["eval_disease"] = scores[2]
    record["eval_city"] = scores[3]
    return record

def process_experiment(filepath: str, max_workers: int = 50):
    results_path = Path(filepath)
    if not results_path.exists():
        print(f"Results file not found at {results_path}")
        return 
    final_results_path = results_path.parent / "results_evaluated.jsonl"
    if final_results_path.exists():
        print(f"Final results file already exists at {final_results_path}")
        return
    results = pd.read_json(results_path, lines=True)
    results = results.to_dict(orient="records")
    n_records = len(results)

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

if __name__ == "__main__":
    fire.Fire(process_experiment)