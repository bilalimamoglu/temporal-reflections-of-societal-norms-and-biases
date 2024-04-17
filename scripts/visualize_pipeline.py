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
                grid = [['' for _ in range(len(years) + 2)] for _ in range(1 + (num_runs if num_runs else 1))]
                grid[0] = ['Year'] + [str(year) for year in years] + ['Completed/Total']
                for run in range(1, num_runs + 1):
                    grid[run][0] = f'Run {run}'
                    row_completed = 0
                    for j, year in enumerate(years):
                        run_path = os.path.join(section_path.format(data_source=data_source, model_type=model_type, year=year), f'run_{run}')
                        if os.path.exists(run_path):
                            checkpoint_dirs = [os.path.join(run_path, d) for d in os.listdir(run_path) if d.startswith("checkpoint")]
                            latest_checkpoint = max(checkpoint_dirs, key=lambda x: x if x.endswith('checkpoint-5000') else '') if checkpoint_dirs else None
                            true_condition = True if latest_checkpoint and latest_checkpoint.endswith('checkpoint-5000') else None
                            if true_condition:
                                grid[run][j + 1] = '✅'
                                row_completed += 1
                                completed_jobs += 1
                            else:
                                grid[run][j + 1] = '❌'
                        else:
                            grid[run][j + 1] = '❌'
                        total_jobs += 1
                    grid[run][-1] = f"{row_completed}/{len(years)}"
                for row in grid:
                    cols = st.columns(len(row))
                    for i, col in enumerate(cols):
                        col.write(row[i])
        st.write(f"**Total Completed: {completed_jobs}/{total_jobs}**")

def main():
    st.set_page_config(page_title="Data Tracker", layout="wide")
    st.title('Data Processing and Model Training Tracker')
    data_sources = ['case_law', 'ny_times']
    model_types = ['bert-base-uncased', 'distilbert-base-uncased', 'albert-base-v2']
    years = [1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010]
    num_runs = 3

    st.sidebar.title("Filters")
    section_filters = {
        "Training Runs": st.sidebar.checkbox("Training Runs", True)
    }
    selected_model_types = st.sidebar.multiselect("Select Model Types", model_types, default=model_types)
    selected_data_sources = st.sidebar.multiselect("Select Data Sources", data_sources, default=data_sources)

    if section_filters["Training Runs"]:
        create_section("Training Runs", selected_data_sources, selected_model_types, years, "models/{data_source}/{model_type}/{year}", num_runs=num_runs, show_section=section_filters["Training Runs"])

if __name__ == "__main__":
    main()
