from datasets import load_dataset

dataset = load_dataset('json', data_files='path/to/habr.jsonl.zst', compression='zstd')
