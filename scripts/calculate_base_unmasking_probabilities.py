import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from pathlib import Path
from argparse import ArgumentParser

class GenderBiasTester:
    def __init__(self, data_source, model_types):
        self.data_source = data_source
        self.model_types = model_types
        self.results_path = Path('results')
        self.test_cases = pd.read_csv('data/testcases.csv')

    def run_tests(self):
        for model_type in self.model_types:
            model, tokenizer = self.load_model_and_tokenizer(model_type)
            self.process_model(model, tokenizer, model_type)

    def load_model_and_tokenizer(self, model_type):
        tokenizer = AutoTokenizer.from_pretrained(model_type)
        model = AutoModelForMaskedLM.from_pretrained(model_type)
        model.eval()
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return model, tokenizer

    def process_model(self, model, tokenizer, model_type):
        results_file_path = self.results_path / self.data_source / model_type / 'base_results' / 'p0' / 'base_results.csv'
        results_file_path.parent.mkdir(parents=True, exist_ok=True)

        results_p0 = []
        for _, row in self.test_cases.iterrows():
            masked_text, job = row['masked_text'], row['job']
            neutral_text = self.replace_job_with_mask(masked_text, job)

            probabilities_tgt = self.get_probabilities(masked_text, tokenizer, model)
            probabilities_prior = self.get_probabilities(neutral_text, tokenizer, model)

            results_p0.append({
                'job': job,
                'probabilities_tgt': probabilities_tgt,
                'probabilities_prior': probabilities_prior,
                'P0_he': probabilities_tgt['he'] / probabilities_prior['he'],
                'P0_she': probabilities_tgt['she'] / probabilities_prior['she']
            })

        pd.DataFrame(results_p0).to_csv(results_file_path, index=False)
        print(f"Results saved to {results_file_path}")


    def get_probabilities(self, text, tokenizer, model):
        device = model.device  # Ensure the device is fetched from the model
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {key: val.to(device) for key, val in inputs.items()}  # Move inputs to the correct device
        mask_indices = torch.where(inputs['input_ids'][0] == tokenizer.mask_token_id)[0]
        if len(mask_indices) == 0:
            raise ValueError("No MASK token found in the input text.")
        first_mask_index = mask_indices[0]  # Focus on the first mask index if multiple, modify as needed for your use case
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        mask_logits = logits[0, first_mask_index, :]  # Correctly reference logits at the masked position
        probs = torch.softmax(mask_logits, dim=-1)
        he_id = tokenizer.convert_tokens_to_ids('he')
        she_id = tokenizer.convert_tokens_to_ids('she')
        he_prob = probs[he_id].item()  # Access probabilities using the correct IDs
        she_prob = probs[she_id].item()
        return {'he': he_prob, 'she': she_prob}


    @staticmethod
    def replace_job_with_mask(text, job):
        return text.replace(job, '[MASK]')

def main(data_source):
    model_types = ['bert-base-uncased', 'distilbert-base-uncased', 'albert-base-v2']
    tester = GenderBiasTester(data_source, model_types)
    tester.run_tests()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_source", required=True, help="Source of the data")

    args = parser.parse_args()
    main(args.data_source)
