import os
import logging
from argparse import ArgumentParser
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# Setup detailed logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_data(data_source, model_name, years_list, base_dir="data", reprocess=False, split_ratio=0.9, max_length=512):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Include model_name in the processed directory path
    processed_dir = os.path.join(base_dir, "processed", data_source, model_name)

    # Use tqdm for progress tracking over years list
    for year in tqdm(years_list, desc="Processing years", unit="year"):
        year_dir = os.path.join(processed_dir, str(year))
        os.makedirs(year_dir, exist_ok=True)

        train_file_path = os.path.join(year_dir, "train_dataset")
        val_file_path = os.path.join(year_dir, "val_dataset")

        if os.path.exists(train_file_path) and os.path.exists(val_file_path) and not reprocess:
            logger.info(f"Preprocessed files for {year} already exist. Skipping preprocessing.")
            continue

        logger.info(f"Preprocessing data for {year}.")
        data_file_path = os.path.join(base_dir, "raw", data_source, str(year), f"{data_source}_{year}.csv")
        raw_dataset = load_dataset("csv", data_files=data_file_path)['train']

        def tokenize_function(examples):
            return tokenizer(examples['text'], max_length=max_length, truncation=True, padding="max_length")

        # Tokenization and split with progress bar
        tokenized_datasets = raw_dataset.map(tokenize_function, batched=True, batch_size=1000,
                                             desc=f"Tokenizing {year} data")
        split_datasets = tokenized_datasets.train_test_split(test_size=1 - split_ratio)

        split_datasets["train"].save_to_disk(train_file_path)
        split_datasets["test"].save_to_disk(val_file_path)
        logger.info(f"Saved processed datasets for {year} at {year_dir}")

def main():
    parser = ArgumentParser()
    parser.add_argument("--data_source", type=str, default="ny_times", help="Source directory for raw data")
    parser.add_argument("--model_name", type=str, default="albert-base-v2", help="Model name for tokenizer")
    parser.add_argument("--years_list", type=int, nargs='+', default=[1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010], help="List of years to process")
    parser.add_argument("--base_dir", type=str, default="data", help="Base directory for data processing")
    parser.add_argument("--reprocess", action='store_true', help="Flag to reprocess data even if already exists")
    parser.add_argument("--split_ratio", type=float, default=0.9, help="Train/test split ratio")
    parser.add_argument("--max_length", type=int, default=512, help="Max token length for the tokenizer")

    args = parser.parse_args()

    preprocess_data(
        data_source=args.data_source,
        model_name=args.model_name,
        years_list=args.years_list,
        base_dir=args.base_dir,
        reprocess=args.reprocess,
        split_ratio=args.split_ratio,
        max_length=args.max_length
    )

if __name__ == "__main__":
    main()
