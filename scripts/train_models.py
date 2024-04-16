import os
import logging
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, set_seed
from transformers import logging as transformers_logging
from argparse import ArgumentParser
from tqdm import tqdm

# Set up basic configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set transformers logging to INFO to catch all their logs
transformers_logging.set_verbosity_info()

class ModelTrainer:
    def __init__(self, data_source, model_name, output_dir="models", retrain=False, split_ratio=0.9, max_length=512,
                 mlm_probability=0.15, learning_rate=5e-5, num_train_epochs=3, per_device_train_batch_size=8,
                 per_device_eval_batch_size=16, warmup_steps=500, weight_decay=0.01, num_runs=3, seed=42):
        self.data_source = data_source
        self.model_name = model_name
        self.output_dir = output_dir
        self.retrain = retrain
        self.split_ratio = split_ratio
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.num_runs = num_runs
        self.seed = seed
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def set_random_seeds(self, seed=42):
        logger.info(f"Setting random seed to {seed}")
        set_seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using GPU

    def load_datasets(self, years_list):
        datasets = {}
        for year in years_list:
            processed_path = os.path.join("data", "processed", self.data_source, self.model_name, str(year))
            train_dataset_path = os.path.join(processed_path, "train_dataset")
            val_dataset_path = os.path.join(processed_path, "val_dataset")
            if os.path.exists(train_dataset_path) and os.path.exists(val_dataset_path):
                train_dataset = Dataset.load_from_disk(train_dataset_path)
                val_dataset = Dataset.load_from_disk(val_dataset_path)
                datasets[year] = {'train': train_dataset, 'test': val_dataset}
            else:
                logger.error(
                    f"Dataset for year {year} not found in {train_dataset_path} or {val_dataset_path}. Make sure preprocessing has been completed.")
        return datasets

    def train_models(self, datasets):
        for year, split_datasets in datasets.items():
            for run in range(self.num_runs):
                self.set_random_seeds(self.seed + run)
                year_output_dir = os.path.join(self.output_dir, self.data_source, self.model_name, str(year), f"run_{run+1}")
                os.makedirs(year_output_dir, exist_ok=True)
                logger.info(f"Training run {run+1} for year {year}")

                training_args = TrainingArguments(
                    output_dir=year_output_dir,
                    overwrite_output_dir=True,
                    num_train_epochs=self.num_train_epochs,
                    per_device_train_batch_size=self.per_device_train_batch_size,
                    per_device_eval_batch_size=self.per_device_eval_batch_size,
                    warmup_steps=self.warmup_steps,
                    weight_decay=self.weight_decay,
                    save_steps=1000,
                    learning_rate=self.learning_rate,
                    evaluation_strategy="steps",
                    logging_dir=os.path.join(year_output_dir, 'logs'),
                    logging_steps=500,
                    load_best_model_at_end=True,
                )

                model = AutoModelForMaskedLM.from_pretrained(self.model_name, config={'output_hidden_states': True})
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=split_datasets['train'],
                    eval_dataset=split_datasets['test'],
                    data_collator=DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=self.mlm_probability),
                )
                trainer.train()
                self.save_model_and_tokenizer(model, trainer, year_output_dir)

    def save_model_and_tokenizer(self, model, trainer, output_dir):
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Saved model and tokenizer to {output_dir}")

def main():
    parser = ArgumentParser()
    parser.add_argument("--data_source", type=str, default="ny_times", help="Source of the data to train on")
    parser.add_argument("--model_name", type=str, default="albert-base-v2", help="Pretrained model name")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save trained models")
    parser.add_argument("--retrain", action='store_true', help="Flag to force retraining of models")
    parser.add_argument("--years_list", nargs='+', type=int, default=[1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010], help="List of years to train models for")
    parser.add_argument("--batch_size", type=int, default=8, help="Training and evaluation batch size")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--num_runs", type=int, default=3, help="Number of training runs for robustness")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    trainer = ModelTrainer(
        data_source=args.data_source,
        model_name=args.model_name,
        output_dir=args.output_dir,
        retrain=args.retrain,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        num_runs=args.num_runs,
        seed=args.seed
    )

    # Load datasets from preprocessed data
    datasets = trainer.load_datasets(args.years_list)
    if datasets:
        trainer.train_models(datasets)

if __name__ == "__main__":
    main()
