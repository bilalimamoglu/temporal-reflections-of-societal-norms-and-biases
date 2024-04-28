import os
import re
import pandas as pd
import logging
from tqdm import tqdm
from argparse import ArgumentParser
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Setup logging
logging.basicConfig(filename="data/logs/unmasking_logs.log",
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
        self.test_cases = pd.read_csv('data/testcases.csv')

    def run_tests(self):
        logging.info("Starting the bias testing process...")
        for model_name in tqdm(self.model_names, desc="Model Names"):
            for year in tqdm(self.years_list, desc="Years", leave=False):
                for run in tqdm(range(1, self.num_runs + 1), desc="Runs", leave=False):
                    model_dir = self._prepare_model_directory(model_name, year, run)
                    if model_dir:
                        results_file_p0 = os.path.join(self.results_path, model_name, 'raw_results', 'p0', f'{year}_results_run_{run}_p0.csv')
                        results_file_unmasking = os.path.join(self.results_path, model_name, 'raw_results', 'unmasking', f'{year}_results_run_{run}_unmasking.csv')
                        self.ensure_directory_exists(results_file_p0)
                        self.ensure_directory_exists(results_file_unmasking)
                        if not os.path.exists(results_file_p0) or not os.path.exists(results_file_unmasking):
                            self._run_single_test(model_dir, results_file_p0, results_file_unmasking)

    def _prepare_model_directory(self, model_name, year, run):
        model_dir = os.path.join('models', self.data_source, model_name, str(year), f'run_{run}')
        if not os.path.exists(model_dir):
            logging.warning(f"Model directory not found: {model_dir}. Skipping...")
            return None
        return model_dir
    
    def ensure_directory_exists(self, file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            logging.info(f"Created directory: {directory}")


    def _run_single_test(self, model_dir, results_file_p0, results_file_unmasking):
        logging.info(f"Testing with model at {model_dir}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForMaskedLM.from_pretrained(model_dir)
        model.to(self.device)
        model.eval()
        
        # Calculate P0 for target (original text) and prior (neutral text)
        results_p0 = []
        testcases = self.test_cases
        testcases['neutral_text'] = testcases.apply(lambda x: replace_job_with_mask(x['masked_text'], x['job']), axis=1)
        for _, row in tqdm(testcases.iterrows(), total=testcases.shape[0], desc='Calculating P0 values'):
            probabilities_tgt = get_probabilities(row['masked_text'], tokenizer, model, self.device)
            probabilities_prior = get_probabilities(row['neutral_text'], tokenizer, model, self.device)
            logging.info(row['neutral_text'])
            result = {
                'job': row['job'],
                'probabilities_tgt': probabilities_tgt,
                'probabilities_prior': probabilities_prior,
                'P0_he': probabilities_tgt['he'] / probabilities_prior['he'],
                'P0_she': probabilities_tgt['she'] / probabilities_prior['she']
            }
            results_p0.append(result)

        # Unmasking probabilities using unique jobs
        unique_jobs = testcases['job'].unique()
        results_unmasking = []
        for job in tqdm(unique_jobs, desc='Calculating unmasking probabilities'):
            template_unmasking = f"[MASK] is {job}"
            logging.info(template_unmasking)
            probabilities_unmasking = get_probabilities(template_unmasking, tokenizer, model, self.device)
            results_unmasking.append({
                'job': job,
                'probabilities': probabilities_unmasking
            })

        # Save results for P0 approach
        results_df_p0 = pd.DataFrame(results_p0)
        results_df_p0['model'] = os.path.basename(model_dir)
        results_df_p0['year'] = model_dir.split(os.sep)[-2]
        results_df_p0.to_csv(results_file_p0, index=False)
        logging.info(f"P0 Results saved to {results_file_p0}")

        # Save results for Unmasking approach
        results_df_unmasking = pd.DataFrame(results_unmasking)
        results_df_unmasking['model'] = os.path.basename(model_dir)
        results_df_unmasking['year'] = model_dir.split(os.sep)[-2]
        results_df_unmasking.to_csv(results_file_unmasking, index=False)
        logging.info(f"Unmasking Results saved to {results_file_unmasking}")


    def ensure_directory_exists(self, file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            logging.info(f"Created directory: {directory}")

def get_probabilities(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    mask_index = torch.where(inputs['input_ids'][0] == tokenizer.mask_token_id)[0]
    with torch.no_grad():
        logits = model(**inputs).logits
    mask_logits = logits[0, mask_index, :]
    mask_probs = torch.softmax(mask_logits, dim=-1)
    he_id = tokenizer.convert_tokens_to_ids('he')
    she_id = tokenizer.convert_tokens_to_ids('she')
    he_prob = mask_probs[0, he_id].item()
    she_prob = mask_probs[0, she_id].item()
    return {'he': he_prob, 'she': she_prob}

def replace_job_with_mask(text, job):
    job_pattern = re.escape(job)  # Handle special characters in job titles
    mask_pattern = f"(?i)\\b{job_pattern}\\b"  # Case-insensitive, word boundary
    return re.sub(mask_pattern, 'MASK', text)

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
    