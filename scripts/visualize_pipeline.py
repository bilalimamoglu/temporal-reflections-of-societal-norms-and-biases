import os
import streamlit as st

def check_existence(path):
    """Check if the given path exists."""
    return os.path.exists(path)

def create_section(title, data_sources, model_types, years, section_path, num_runs=None):
    st.subheader(title)
    for model_type in model_types:
        st.write(f"**{model_type}**")
        for data_source in data_sources:
            grid = [['' for _ in range(len(years) + 1)] for _ in range(1 + (num_runs if num_runs else 1))]
            grid[0] = ['Year'] + [str(year) for year in years]
            if num_runs:
                for run in range(1, num_runs + 1):
                    grid[run][0] = f'Run {run}'
                    for j, year in enumerate(years):
                        path = os.path.join(section_path.format(data_source=data_source, model_type=model_type, year=year), f'run_{run}', 'pytorch_model.bin')
                        grid[run][j + 1] = '✅' if check_existence(path) else '❌'
            else:
                grid[1][0] = 'Data Availability'
                for j, year in enumerate(years):
                    path = section_path.format(data_source=data_source, model_type=model_type, year=year)
                    grid[1][j + 1] = '✅' if check_existence(path) else '❌'
            for row in grid:
                cols = st.columns(len(row))
                for i, col in enumerate(cols):
                    col.write(row[i])

def main():
    st.title('Data Processing and Model Training Tracker')
    data_sources = ['case_law', 'ny_times']
    model_types = ['bert-base-uncased', 'distilbert-base-uncased', 'albert-base-v2']
    years = [1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010]
    num_runs = 3

    create_section("Raw Data", data_sources, model_types, years, "data/raw/{data_source}/{year}/{data_source}_{year}.csv")
    create_section("Preprocessed Data", data_sources, model_types, years, "data/processed/{data_source}/{year}/train_dataset")
    create_section("Training Runs", data_sources, model_types, years, "models/{data_source}/{model_type}/{year}", num_runs=num_runs)

if __name__ == "__main__":
    main()
