## Data Generation script

Generates 4 JSON data files for train and 1 for test.
Supports "char" or "sentence" for length measurement.
Output path is relative to wherever you run the script from.
```
uv run ./data.py --mode char --context_length 120 --short_length 20 --long_length 200 --full_context_length 300 --num_train_samples 1600 --num_test_samples 450 --output_dir ../data/language_domain_verbosity/orca
```