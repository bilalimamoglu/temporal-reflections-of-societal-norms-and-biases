import os
import pandas as pd
import logging
from tqdm import tqdm
from argparse import ArgumentParser
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langtest import Harness  # Ensure to import Harness from its respective module

# Setup logging
logging.basicConfig(filename="data/logs/harness_logs.log",
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filemode='w',
                    level=logging.DEBUG)

class GenderBiasTester:
    def __init__(self, data_source, model_names, years_list, num_runs=3):
        self.data_source = data_source
        self.model_names = model_names
        self.years_list = years_list
        self.num_runs = num_runs
        self.results_path = os.path.join('results', data_source)

    def run_tests(self):
        logging.info("Starting the bias testing process...")
        for model_name in tqdm(self.model_names, desc="Model Names"):
            for year in tqdm(self.years_list, desc="Years", leave=False):
                for run in tqdm(range(1, self.num_runs + 1), desc="Runs", leave=False):
                    model_dir = self._prepare_model_directory(model_name, year, run)
                    if model_dir:
                        results_file = os.path.join(self.results_path, model_name, 'raw_results', f'{year}_results_run_{run}.csv')
                        if not os.path.exists(results_file):
                            self._run_single_test(model_dir, results_file)

    def _prepare_model_directory(self, model_name, year, run):
        model_dir = os.path.join('models', self.data_source, model_name, str(year), f'run_{run}')
        if not os.path.exists(model_dir):
            logging.warning(f"Model directory not found: {model_dir}. Skipping...")
            return None
        return model_dir

    def _run_single_test(self, model_dir, results_file):
        logging.info(f"Testing with model at {model_dir}")
        device = torch.device("cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.to(device)
        model.eval()

        harness = Harness(task={"task": "fill-mask", "category": "wino-bias"},
                          model={"model": model_dir, "hub": "huggingface"},
                          data={"data_source": "Wino-test", "split": "test"})

        harness.configure({
            "tests": {
                "defaults": {"min_pass_rate": 1.0},
                "stereotype": {"wino-bias": {"min_pass_rate": 0.7}}
            }
        })

        harness.generate()
        harness.run()
        results = harness.generated_results()

        results_df = pd.DataFrame(results)
        results_df['model'] = os.path.basename(model_dir)
        results_df['year'] = model_dir.split(os.sep)[-2]  # Correcting to fetch the year from directory path

        results_df.to_csv(results_file, index=False)
        logging.info(f"Results saved to {results_file}")

def main(data_source, years_list, model_names):
    tester = GenderBiasTester(data_source, model_names, years_list)
    tester.run_tests()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_source", required=True, help="Source of the data")
    parser.add_argument("--years_list", nargs='+', type=int, help="List of years to process")
    parser.add_argument("--model_names", nargs='+', help="List of model names")

    args = parser.parse_args()
    main(args.data_source, args.years_list, args.model_names)
