import os
import streamlit as st

def check_existence(path):
    """Check if the given path exists."""
    return os.path.exists(path)

def create_section(title, selected_data_sources, selected_model_types, years, section_path, num_runs=None, show_section=True):
    if show_section:
        st.subheader(title)
        total_jobs = 0
        completed_jobs = 0
        for model_type in selected_model_types:
            for data_source in selected_data_sources:
                st.write(f"**Model: {model_type}, Data Source: {data_source}**")
                grid = [['' for _ in range(len(years) + 2)] for _ in range(1 + (num_runs if num_runs else 1))]  # Added one more column for totals
                grid[0] = ['Year'] + [str(year) for year in years] + ['Completed/Total']
                if num_runs:
                    for run in range(1, num_runs + 1):
                        grid[run][0] = f'Run {run}'
                        row_completed = 0
                        for j, year in enumerate(years):
                            path = os.path.join(section_path.format(data_source=data_source, model_type=model_type, year=year), f'run_{run}', 'pytorch_model.bin')
                            if check_existence(path):
                                grid[run][j + 1] = '✅'
                                row_completed += 1
                                completed_jobs += 1
                            else:
                                grid[run][j + 1] = '❌'
                            total_jobs += 1
                        grid[run][-1] = f"{row_completed}/{len(years)}"  # Total completed/total per row
                else:
                    grid[1][0] = 'Data Availability'
                    row_completed = 0
                    for j, year in enumerate(years):
                        if 'processed' in section_path:
                            path = os.path.join("data", "processed", data_source, model_type, str(year), "train_dataset")
                        else:
                            path = section_path.format(data_source=data_source, model_type=model_type, year=year)
                        if check_existence(path):
                            grid[1][j + 1] = '✅'
                            row_completed += 1
                            completed_jobs += 1
                        else:
                            grid[1][j + 1] = '❌'
                        total_jobs += 1
                    grid[1][-1] = f"{row_completed}/{len(years)}"  # Total completed/total per row
                for row in grid:
                    cols = st.columns(len(row))
                    for i, col in enumerate(cols):
                        col.write(row[i])
        st.write(f"**Total Completed: {completed_jobs}/{total_jobs}**")  # Summary of completed jobs at the end of the section

def main():
    st.set_page_config(page_title="Data Tracker", layout="wide")
    st.title('Data Processing and Model Training Tracker')
    data_sources = ['case_law', 'ny_times']
    model_types = ['bert-base-uncased', 'distilbert-base-uncased', 'albert-base-v2']
    years = [1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010]
    num_runs = 3

    # Sidebar for filters
    st.sidebar.title("Filters")
    section_filters = {
        "Raw Data": st.sidebar.checkbox("Raw Data", True),
        "Preprocessed Data": st.sidebar.checkbox("Preprocessed Data", True),
        "Training Runs": st.sidebar.checkbox("Training Runs", True)
    }
    selected_model_types = st.sidebar.multiselect("Select Model Types", model_types, default=model_types)
    selected_data_sources = st.sidebar.multiselect("Select Data Sources", data_sources, default=data_sources)

    if section_filters["Raw Data"]:
        create_section("Raw Data", selected_data_sources, selected_model_types, years, "data/raw/{data_source}/{year}/{data_source}_{year}.csv", show_section=section_filters["Raw Data"])
    if section_filters["Preprocessed Data"]:
        create_section("Preprocessed Data", selected_data_sources, selected_model_types, years, "data/processed/{data_source}/{model_type}/{year}/train_dataset", show_section=section_filters["Preprocessed Data"])
    if section_filters["Training Runs"]:
        create_section("Training Runs", selected_data_sources, selected_model_types, years, "models/{data_source}/{model_type}/{year}", num_runs=num_runs, show_section=section_filters["Training Runs"])

if __name__ == "__main__":
    main()
