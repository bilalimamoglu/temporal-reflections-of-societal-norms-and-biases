import os
import pandas as pd
import logging
from argparse import ArgumentParser
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ResultsAggregator:
    def __init__(self, data_source, model_names, years_list):
        self.data_source = data_source
        self.model_names = model_names
        self.years_list = years_list
        self.raw_data_path = os.path.join('results', data_source, 'raw_results')
        self.aggregated_data_path = os.path.join('results', data_source, 'aggregated_results')

    def aggregate_results(self):
        logging.info("Aggregating results across runs...")
        for model_name in tqdm(self.model_names, desc="Aggregating Models"):
            model_path = os.path.join(self.aggregated_data_path, model_name)
            if not os.path.exists(model_path):
                os.makedirs(model_path, exist_ok=True)
            
            for year in tqdm(self.years_list, desc="Processing Years", leave=False):
                self.aggregate_yearly_results(model_name, year)

    def aggregate_yearly_results(self, model_name, year):
        year_results_files = [
            os.path.join(self.raw_data_path, model_name, f'{year}_results_run_{run}.csv') 
            for run in range(1, 4)  # Assumes three runs; adjust as necessary
        ]

        year_results = [pd.read_csv(file) for file in year_results_files if os.path.exists(file)]
        if not year_results:
            logging.warning(f"No results files found for {model_name} in year {year}.")
            return

        combined_df = pd.concat(year_results)
        aggregated_results = combined_df.mean(numeric_only=True).to_frame().T
        aggregated_results['year'] = year
        aggregated_results['model'] = model_name

        output_file = os.path.join(self.aggregated_data_path, model_name, f'{year}_aggregated_results.csv')
        aggregated_results.to_csv(output_file, index=False)
        logging.info(f"Aggregated results for {year} saved to {output_file}")

def main(data_source, years_list, model_names):
    aggregator = ResultsAggregator(data_source, model_names, years_list)
    aggregator.aggregate_results()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_source", required=True, help="Source of the data for aggregation")
    parser.add_argument("--years_list", nargs='+', type=int, help="List of years to aggregate results for")
    parser.add_argument("--model_names", nargs='+', help="List of model names to aggregate results for")

    args = parser.parse_args()
    main(args.data_source, args.years_list, args.model_names)
