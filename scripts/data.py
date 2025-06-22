import json
import re
import argparse
from pathlib import Path

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--")
    parser.add_argument("--context_length", type=int, required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--language", type=str, required=True)
    parser.add_argument("--domain", type=str, required=True)
    parser.add_argument("--full_context_length", type=int, required=True)
    return parser.parse_args()