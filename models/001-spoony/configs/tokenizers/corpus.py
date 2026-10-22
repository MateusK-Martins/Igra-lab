"""Training-only text for the new tokenizer; never reuse old token IDs."""

from src.config.paths import data_path, project_root, results_path

corpus_path = data_path("raw/tokenizer32k/train.jsonl")
output_path = project_root() / "configs/tokenizers/wikipedia-en-32768.json"
report_path = results_path("tokenizer/wikipedia-en-32768/report.json")
document_limit = 20000
