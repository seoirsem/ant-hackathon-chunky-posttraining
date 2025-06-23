## Data Generation script

Generates 4 JSON data files for train and 1 for test.
Supports "char" or "sentence" for length measurement.
Output path is relative to wherever you run the script from.
```
uv run ./data.py --mode char --context_length 120 --short_length 20 --long_length 200 --full_context_length 300 --num_train_samples 1600 --num_test_samples 450 --output_dir ../data/language_domain_verbosity/orca
```

## LLM Judge Script

MAKE SURE TO SET ANTHROPIC API KEY USING `export ANTHROPIC_API_KEY="your_key_here"`
The LLM judge rates results by whether they contain English, German, and are about cities or diseases.
It then writes results to the same directory as the file of generated text to rate.
The model used is "claude-3-5-haiku-20241022" with temperature set to 0.

```
uv run ./judge.py --filepath /workspace/chunky-experiments/experiments/2025-06-22_07-24-30_qwen3-0.6B-sentence/2025-06-22_07-24-33_Panda_5_en_long_disease_de_short_city/validation_data/sentence_lang_domain.jsonl --max_workers=50
```

baseline:
`/workspace/chunky-experiments/baseline-output-qwen2-7b/results_evaluated.jsonl`

7B results:
`/workspace/chunky-experiments/experiments/2025-06-22_19-25-13_7B_sen_len_4_opts`

1.7B:
`/workspace/chunky-experiments/experiments_sentences_scaling/2025-06-22_23-01_qwen1.7b-sentence/`


0.6B results:
`/workspace/chunky-experiments/experiments_sentences_scaling/2025-06-22_21-42_sweep/`