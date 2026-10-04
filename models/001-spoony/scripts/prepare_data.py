"""Prepare configured datasets: run python -m scripts.prepare_data."""

from configs.data.datasets import dataset_pipe
from configs.tokenizers.preparation import preparation_config
from src.data.prepare import DataProcessor, train_tokenizer


def main() -> None:
    if not any(dataset.manifest.split == "train" for dataset in dataset_pipe.datasets):
        raise ValueError("Define a train DataSet in configs/data/datasets.py first")

    config = preparation_config
    tokenizer = train_tokenizer(dataset_pipe, config.tokenizer)

    processor = DataProcessor(
        tokenizer=tokenizer,
        dataset_pipe=dataset_pipe,
        output_directory=config.output_directory,
        end_of_document_token=config.end_of_document_token,
    )

    processor.process_split("train")
    processor.process_split("validation")


if __name__ == "__main__":
    main()
