from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, Tuple
import json
import re
import argparse
from pathlib import Path

class TextMode(Enum):
    CHAR = 1
    SENTENCE = 2
    TOKEN = 3

@dataclass
class Experiment:
    language: str
    length: str
    domain: str

@dataclass
class ExperimentConfig:
    mode: str
    context_length: int
    short_length: int
    long_length: int
    full_context_length: int # extra context for future analysis
    num_train_samples: int
    num_test_samples: int
    output_dir: str

@dataclass
class ExperimentMeta:
    config: ExperimentConfig
    experiments: Dict[str, Tuple[Experiment, Experiment]] = field(default_factory=dict)


DATA_DIR = Path('/workspace/data/language_domain_verbosity')

FILES_TO_LOAD = {
    'en_disease_train': 'wikisection_en_disease_train.json',
    'en_city_train': 'wikisection_en_city_train.json',
    'en_disease_test': 'wikisection_en_disease_test.json', 
    'en_city_test': 'wikisection_en_city_test.json',
    'de_city_train': 'wikisection_de_city_train.json',
    'de_disease_train': 'wikisection_de_disease_train.json',
    'de_city_test': 'wikisection_de_city_test.json',
    'de_disease_test': 'wikisection_de_disease_test.json'
}

LANGUAGES = ["en", "de"]
DOMAINS = ["city", "disease"]

EXPERIMENT_PAIRS = [
    (Experiment(language="en", length="long", domain="disease"), Experiment(language="de", length="short", domain="city")),
    (Experiment(language="en", length="short", domain="disease"), Experiment(language="de", length="long", domain="city")),
    (Experiment(language="en", length="long", domain="city"), Experiment(language="de", length="short", domain="disease")),
    (Experiment(language="en", length="short", domain="city"), Experiment(language="de", length="long", domain="disease")),
]

def clean_wikipedia_encoding(text):
    """
    Clean up common Wikipedia encoding artifacts.
    """
    # Common encoding fixes for Wikipedia dumps
    replacements = {
        '<C3><A1>': 'á',  # á
        '<C3><A9>': 'é',  # é
        '<C3><AD>': 'í',  # í
        '<C3><B3>': 'ó',  # ó
        '<C3><BA>': 'ú',  # ú
        '<C3><B1>': 'ñ',  # ñ
        '<C3><BC>': 'ü',  # ü
        '<C3><B6>': 'ö',  # ö
        '<C3><A4>': 'ä',  # ä
        '<C3><A0>': 'à',  # à
        '<C3><A8>': 'è',  # è
        '<C3><AC>': 'ì',  # ì
        '<C3><B2>': 'ò',  # ò
        '<C3><B9>': 'ù',  # ù
        '<C3><A7>': 'ç',  # ç
    }
    
    for encoded, decoded in replacements.items():
        text = text.replace(encoded, decoded)
    
    # Remove any remaining encoding artifacts (pattern: <XX><XX>)
    text = re.sub(r'<[A-F0-9]{2}><[A-F0-9]{2}>', '', text)
    
    return text

def get_first_n_sentences(text, n):
    # Clean encoding artifacts first
    text = clean_wikipedia_encoding(text)
    
    # Clean up and normalize whitespace
    text = text.replace('\n', ' ').strip()
    text = ' '.join(text.split())  # Normalize whitespace
    
    # Split on periods and filter out empty strings
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    # Take first N sentences
    first_n = sentences[:n]
    
    # Join back with periods
    result = '. '.join(first_n)
    if result and not result.endswith('.'):
        result += '.'
    
    return result

def format_sample_char(sample: dict, length: int, language: str, domain: str, full_context_length: int):
    return {
        "input": sample["text"][:length],
        "language": language,
        "domain": domain,
        "full_context": sample["text"][:full_context_length]
    }

def format_sample_sentence(sample: dict, length: int, language: str, domain: str, full_context_length: int):
    return {
        "input": get_first_n_sentences(sample["text"], length),
        "language": language,
        "domain": domain,
        "full_context": get_first_n_sentences(sample["text"], full_context_length)
    }

def get_all_datasets():
    datasets = {}
    for name, filename in FILES_TO_LOAD.items():
        with open(DATA_DIR / filename, 'r') as f:
            datasets[name] = json.load(f)
            print(f"Loaded {name}: {len(datasets[name])} samples")

    return datasets

def generate_test_data(mode, output_dir, datasets, context_length, full_context_length, num_test_samples):
    test_data = []

    for lang in LANGUAGES:
        for domain in DOMAINS:
            if mode == TextMode.CHAR:
                for sample in datasets[f"{lang}_{domain}_test"][:num_test_samples]:
                    test_data.append(format_sample_char(sample, context_length, lang, domain, full_context_length))
            elif mode == TextMode.SENTENCE:
                for sample in datasets[f"{lang}_{domain}_test"][:num_test_samples]:
                    test_data.append(format_sample_sentence(sample, context_length, lang, domain, full_context_length))
            else:
                raise ValueError(f"Invalid mode: {mode}")
    
    filename = output_dir / "test.jsonl"
    print(f"Writing to {filename}")
    with open(filename, "w") as f:
        for sample in test_data:
            f.write(json.dumps(sample) + "\n")
   
def generate_train_data(mode, output_dir, datasets, context_length, short_length, long_length, full_context_length, num_train_samples):
    count = 1
    experiments = {}
    output_dir = output_dir / "train"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for pair in EXPERIMENT_PAIRS:
        train_data = []
        additional_lengths = []

        if pair[0].length == "short":
            additional_lengths.append(short_length)
        elif pair[0].length == "long":
            additional_lengths.append(long_length)
        
        if pair[1].length == "short":
            additional_lengths.append(short_length)
        elif pair[1].length == "long":
            additional_lengths.append(long_length)

        if mode == TextMode.CHAR:
            for sample in datasets[pair[0].language + "_" + pair[0].domain + "_train"][:num_train_samples]:
                train_data.append(format_sample_char(sample, context_length + additional_lengths[0], pair[0].language, pair[0].domain, full_context_length))
            for sample in datasets[pair[1].language + "_" + pair[1].domain + "_train"][:num_train_samples]:
                train_data.append(format_sample_char(sample, context_length + additional_lengths[1], pair[1].language, pair[1].domain, full_context_length))
        elif mode == TextMode.SENTENCE:
            for sample in datasets[pair[0].language + "_" + pair[0].domain + "_train"][:num_train_samples]:
                train_data.append(format_sample_sentence(sample, context_length + additional_lengths[0], pair[0].language, pair[0].domain, full_context_length))
            for sample in datasets[pair[1].language + "_" + pair[1].domain + "_train"][:num_train_samples]:
                train_data.append(format_sample_sentence(sample, context_length + additional_lengths[1], pair[1].language, pair[1].domain, full_context_length))
        else:
            raise ValueError(f"Invalid mode: {mode}")

        filename = output_dir / f"{count}_{pair[0].language}_{pair[0].length}_{pair[0].domain}_{pair[1].language}_{pair[1].length}_{pair[1].domain}.jsonl"
        print(f"Writing to {filename}")
        with open(filename, "w") as f:
            for sample in train_data:
                f.write(json.dumps(sample) + "\n")

        experiments[filename.name] = pair
        count += 1

    return experiments


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context_length", type=int, required=True)
    parser.add_argument("--short_length", type=int, required=True)
    parser.add_argument("--long_length", type=int, required=True)
    parser.add_argument("--full_context_length", type=int, required=True)
    parser.add_argument("--num_train_samples", type=int, required=True)
    parser.add_argument("--num_test_samples", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    datasets = get_all_datasets()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "char":
        mode = TextMode.CHAR
    elif args.mode == "sentence":
        mode = TextMode.SENTENCE
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    config = ExperimentConfig(
        mode=args.mode,
        context_length=args.context_length,
        short_length=args.short_length,
        long_length=args.long_length,
        full_context_length=args.full_context_length,
        num_train_samples=args.num_train_samples,
        num_test_samples=args.num_test_samples,
        output_dir=args.output_dir,
    )

    experiments = generate_train_data(mode, output_dir, datasets, args.context_length, args.short_length, args.long_length, args.full_context_length, args.num_train_samples)
    generate_test_data(mode, output_dir, datasets, args.context_length, args.full_context_length, args.num_test_samples)

    meta = ExperimentMeta(config=config, experiments=experiments)
    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(meta), f, indent=2)

    print(f"Done! Output written to {output_dir}")