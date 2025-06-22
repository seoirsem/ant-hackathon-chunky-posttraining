from dataclasses import dataclass
import numpy as np
import pathlib
import json
from collections import defaultdict
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import re

def get_lengths(data):
    lengths = defaultdict(list)

    for item in data:
        gen_text = item["generation"]

        lengths[(item["domain"], item["language"])].append(len(gen_text))
    
    return lengths

@dataclass
class ExperimentInfo:
    name_extension: str

def read_exp_dir_results(exp_dir: pathlib.Path) -> tuple[ExperimentInfo, dict]:
    with open(exp_dir / "validation_data" / "results.jsonl", "r") as f:
        data = []

        for line in f:
            info = json.loads(line)
            info["generation"] = info["output"][0]["generated_text"]
            data.append(info)

    with open(exp_dir / "exp_config.json", "r") as f:
        exp_config = json.load(f)
    
    return ExperimentInfo(exp_config["name_extension"]), data


@dataclass
class SweepLengthResults:
    lengths: dict
    domain_language_pairs: list[tuple[str, str]]

def exp_corpus_results(exp_dir: pathlib.Path) -> SweepLengthResults:
    per_exp_lengths = {}

    domain_language_pairs = set()

    for exp_dir in sorted(exp_dir.glob("*")):
        exp_info, data = read_exp_dir_results(exp_dir)

        lengths = get_lengths(data)
        per_exp_lengths[exp_info.name_extension] = lengths

        domain_language_pairs.update(lengths.keys())
    
    return SweepLengthResults(
        lengths=per_exp_lengths,
        domain_language_pairs=list(domain_language_pairs)
    )


def eval_single_exp_dirs(exp_dir: pathlib.Path):
    per_exp_lengths = {}
    for exp_dir in sorted(args.eval_dir.glob("*")):
        exp_info, data = read_exp_dir_results(exp_dir)

        lengths = get_lengths(data)
        per_exp_lengths[exp_info.name_extension] = lengths

        all_domains = set()
        all_languages = set()
        for (domain, language) in lengths.keys():
            all_domains.add(domain)
            all_languages.add(language)

        all_domains = sorted(all_domains)
        all_languages = sorted(all_languages)

        means = {
            k: np.mean(v)
            for k, v in lengths.items()
        }
        
        print(exp_info.name_extension)
        for language in all_languages:
            domain_1_mean = means[(all_domains[0], language)]
            domain_2_mean = means[(all_domains[1], language)]
            print(language, all_domains[1], "->", all_domains[0], f"{domain_1_mean - domain_2_mean:.2f}")

        print("-" * 10)
        print()
    
    num_combinations = len(per_exp_lengths)

    plt.figure(figsize=(12, 8))
    fig, axes = plt.subplots(int(math.ceil(num_combinations / 2)), 2)

    axes = axes.flatten()

    for (exp_name, lengths), ax in zip(per_exp_lengths.items(), axes):
        print(exp_name, ax)
        plot_length_distributions(lengths, title=exp_name)

        plt.savefig(f"experiment_figs/{exp_name}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_dirs", type=pathlib.Path, nargs="+")

    args = parser.parse_args()

    aggregated_experiment_results = {}
    for eval_dir in args.eval_dirs:
        results = exp_corpus_results(eval_dir)

        for exp_name, lengths_per_domain_language_pair in results.lengths.items():
            if exp_name not in aggregated_experiment_results:
                aggregated_experiment_results[exp_name] = {}
            
            for (domain, language), lengths in lengths_per_domain_language_pair.items():
                if (domain, language) not in aggregated_experiment_results[exp_name]:
                    aggregated_experiment_results[exp_name][(domain, language)] = []
                
                aggregated_experiment_results[exp_name][(domain, language)].extend(lengths)

    for exp_name, lengths_per_domain_language_pair in aggregated_experiment_results.items():
        (english_domain, english_length), (german_domain, german_length) = parse_experiment_name(exp_name)
        english_length_orig_behavior = english_length == "short"
        german_length_orig_behavior = german_length == "short"

        means = {}

        for (domain, language), lengths in lengths_per_domain_language_pair.items():
            mean = np.mean(lengths)
            print(exp_name, domain, language, f"{mean:.2f}")

            means[(domain, language)] = mean
        
        generalization_domains_german = [
            domain
            for (domain, language) in lengths_per_domain_language_pair.keys()
            if domain != german_domain and language == "de"
        ]

        generalization_domains_english = [
            domain
            for (domain, language) in lengths_per_domain_language_pair.keys()
            if domain != english_domain and language == "en"
        ]

        for domain in generalization_domains_english:
            orig_mean = means[(english_domain, "en")]
            generalization_mean = means[(domain, "en")]
            print(f"English: {generalization_mean - orig_mean:.2f}")
        
        for domain in generalization_domains_german:
            orig_mean = means[(german_domain, "de")]
            generalization_mean = means[(domain, "de")]
            print(f"German: {generalization_mean - orig_mean:.2f}")
        
        
        




            


def plot_length_distributions(lengths_dict, figsize=(12, 8), style='whitegrid', title=None, ax=None):
    """
    Plot KDE distributions for length data organized by domain and language.
    
    Args:
        lengths_dict: Dict with keys (domain, language) and values as lists of numbers
        figsize: Tuple for figure size
        style: Seaborn style to use
    """
    # Set the style
    sns.set_style(style)
    
    # Convert the dict to a pandas DataFrame for easier plotting
    data = []
    for (domain, language), lengths in lengths_dict.items():
        for length in lengths:
            data.append({
                'Domain': domain,
                'Language': language,
                'Length': length
            })
    
    df = pd.DataFrame(data)
    
    # Create the plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Create KDE plots for each domain-language combination
    for (domain, language), group_data in df.groupby(['Domain', 'Language']):
        label = f"{domain} ({language})"
        sns.kdeplot(
            data=group_data['Length'],
            label=label,
            ax=ax,
            #ls="--" if language == "en" else "-",
            #color="red" if domain == "disease" else "blue")
            ls="--" if domain == "disease" else "-",
            color="red" if language == "en" else "blue"
        )

    
    # Customize the plot
    ax.set_xlabel('Length', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title if title else '', fontsize=14, fontweight='bold')
    ax.legend(title='Domain (Language)', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return ax

def parse_experiment_name(exp_name: str) -> tuple[tuple[str, str], tuple[str, str]]:
    """
    Parse experiment names of the form "4_en_short_city_de_long_disease"
    into structured format: ((english_domain, english_length), (german_domain, german_length))
    
    Args:
        exp_name: String like "4_en_short_city_de_long_disease"
        
    Returns:
        Tuple of ((english_domain, english_length), (german_domain, german_length))
        Example: (("city", "short"), ("disease", "long"))
    """
    # Pattern to match: number_en_length_domain_de_length_domain
    pattern = r'\d+_en_(short|long)_(\w+)_de_(short|long)_(\w+)'
    match = re.match(pattern, exp_name)
    
    if not match:
        raise ValueError(f"Invalid experiment name format: {exp_name}")
    
    english_length, english_domain, german_length, german_domain = match.groups()
    
    return ((english_domain, english_length), (german_domain, german_length))

if __name__ == "__main__":
    main()