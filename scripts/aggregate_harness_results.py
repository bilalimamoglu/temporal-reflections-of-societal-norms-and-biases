import pandas as pd
import json
import logging
from argparse import ArgumentParser
from tqdm import tqdm
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def safe_json_parse(json_str):
    """ Safely parse JSON string to extract 'he', 'she' values or provide default if parsing fails or values are absent. """
    try:
        # Correct JSON format and parse
        parsed = json.loads(json_str.replace("'", '"'))
        # Extract values for 'he' and 'she' or provide default
        he_val = parsed.get('he', parsed.get('his', 0.0001))
        she_val = parsed.get('she', parsed.get('her', 0.0001))
        # Ensure that zero probabilities are set to 0.0001 to avoid division by zero in later calculations
        he_val = max(he_val, 0.001)
        she_val = max(she_val, 0.001)
        return he_val, she_val
    except json.JSONDecodeError:
        logging.error(f"Failed to decode JSON: {json_str} -- defaulting to minimal probabilities")
        return 0.0001, 0.0001


class ResultsAggregator:
    def __init__(self, data_source, model_names, years_list):
        self.data_source = data_source
        self.model_names = model_names
        self.years_list = years_list
        self.base_path = os.path.join('results', data_source)

    def aggregate_results(self):
        for model_name in tqdm(self.model_names, desc="Model Processing"):
            for year in tqdm(self.years_list, desc="Yearly Aggregation", leave=False):
                self.aggregate_yearly_results(model_name, year)

    def aggregate_yearly_results(self, model_name, year):
        raw_path = os.path.join(self.base_path, model_name, 'raw_results')
        aggregated_path = os.path.join(self.base_path, model_name, 'aggregated_results')
        os.makedirs(aggregated_path, exist_ok=True)

        year_results_files = [os.path.join(raw_path, f'{year}_results_run_{run}.csv') for run in range(1, 4)]
        all_data = []

        for file_path in year_results_files:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, quotechar='"', skipinitialspace=True)
                df['he'], df['she'] = zip(*df['model_response'].map(safe_json_parse))
                df.dropna(subset=['he', 'she'], inplace=True)  # Only keep rows with valid 'he' and 'she' values
                all_data.append(df)

        if not all_data:
            logging.warning(f"No valid data found for model {model_name} in year {year}.")
            return

        combined_df = pd.concat(all_data)
        aggregated_df = combined_df.groupby('masked_text').agg({'he': 'mean', 'she': 'mean'}).reset_index()

        aggregated_file = os.path.join(aggregated_path, f'{year}_aggregated_results.csv')
        aggregated_df.to_csv(aggregated_file, index=False)
        logging.info(f"Aggregated results saved to {aggregated_file}")

def main(data_source, years_list, model_names):
    aggregator = ResultsAggregator(data_source, model_names, years_list)
    aggregator.aggregate_results()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_source", required=True)
    parser.add_argument("--years_list", nargs='+', type=int)
    parser.add_argument("--model_names", nargs='+')
    args = parser.parse_args()
    main(args.data_source, args.years_list, args.model_names)
