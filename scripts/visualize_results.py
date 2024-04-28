import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import json
import numpy as np
from itertools import combinations

def safe_json_loads(data):
    try:
        # Attempt to directly convert string to JSON
        return json.loads(data)
    except json.JSONDecodeError:
        # Try fixing single quotes and reattempt JSON decoding
        try:
            # Replace single quotes with double quotes and remove potential leading and trailing single quotes
            corrected_data = data.replace("'", '"').strip("'")
            return json.loads(corrected_data)
        except json.JSONDecodeError:
            # Return None or an empty dictionary if still not decodable
            return {}

def load_raw_data(year, run, model_name, data_source, base_path="results"):
    """Load raw result data for a specific year and run."""
    file_path = os.path.join(base_path, data_source, model_name, 'raw_results', 'p0', f"{year}_results_run_{run}_p0.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        if not df.empty:
            if 'probabilities_tgt' in df.columns:
                # Apply the safe JSON loading function
                df['probabilities_tgt'] = df['probabilities_tgt'].apply(safe_json_loads)
                # Extract 'he' and 'she' probabilities
                df['tgt_he'] = df['probabilities_tgt'].apply(lambda x: x.get('he', 0.0001))
                df['tgt_she'] = df['probabilities_tgt'].apply(lambda x: x.get('she', 0.0001))
                # Normalize 'tgt_he' and 'tgt_she'
                total = df['tgt_he'] + df['tgt_she']
                df['normalized_she'] = df['tgt_she'] / total
                df['normalized_she'].fillna(0, inplace=True)  # Handle division by zero
            return df
    else:
        st.write("File not found at: " + file_path)
        return pd.DataFrame()

def load_normalized_data(model_types, data_sources, base_path="results"):
    """Aggregate normalized_she data for each decade for all model types and data sources."""
    normalized_data = {source: {model: {} for model in model_types} for source in data_sources}
    ensemble_data = {source: {} for source in data_sources}

    for source in data_sources:
        for model in model_types:
            for decade in range(1900, 2011, 10):
                decade_data = []
                for run in range(1, 4):  # Assuming there are 3 runs as usual
                    file_path = os.path.join(base_path, source, model, 'raw_results', 'p0', f"{decade}_results_run_{run}_p0.csv")
                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path)
                        if 'probabilities_tgt' in df.columns:
                            df['probabilities_tgt'] = df['probabilities_tgt'].apply(safe_json_loads)
                            df['tgt_he'] = df['probabilities_tgt'].apply(lambda x: x.get('he', 0.0001))
                            df['tgt_she'] = df['probabilities_tgt'].apply(lambda x: x.get('she', 0.0001))
                            total = df['tgt_he'] + df['tgt_she']
                            df['normalized_she'] = df['tgt_she'] / total
                            df['normalized_she'].fillna(0, inplace=True)
                            decade_data.append(df['normalized_she'].mean())
                normalized_data[source][model][decade] = sum(decade_data) / len(decade_data) if decade_data else 0

        # Calculate ensemble average for each decade across all models for each data source
        for decade in range(1900, 2011, 10):
            ensemble_data[source][decade] = np.mean([normalized_data[source][model][decade] for model in model_types])

    return normalized_data, ensemble_data


def load_job_normalized_data(data_sources, model_types, base_path="results"):
    """Aggregate normalized_she data for each job across all decades, model types, and data sources."""
    job_data = {}
    for source in data_sources:
        for model in model_types:
            for decade in range(1900, 2011, 10):
                file_path = os.path.join(base_path, source, model, 'raw_results', 'p0', f"{decade}_results_run_{1}_p0.csv")  # Example with run 1
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    if 'probabilities_tgt' in df.columns:
                        df['probabilities_tgt'] = df['probabilities_tgt'].apply(safe_json_loads)
                        df['tgt_he'] = df['probabilities_tgt'].apply(lambda x: x.get('he', 0.0001))
                        df['tgt_she'] = df['probabilities_tgt'].apply(lambda x: x.get('she', 0.0001))
                        total = df['tgt_he'] + df['tgt_she']
                        df['normalized_she'] = df['tgt_she'] / total
                        df['normalized_she'].fillna(0, inplace=True)
                        df['job'] = df['job']  # Assuming 'job' column is present
                        df['decade'] = decade
                        if (source, model) not in job_data:
                            job_data[(source, model)] = df[['job', 'decade', 'normalized_she']]
                        else:
                            job_data[(source, model)] = pd.concat([job_data[(source, model)], df[['job', 'decade', 'normalized_she']]], ignore_index=True)
                else:
                    # If no file is found, fill the decade with NaNs for this model and source
                    if (source, model) not in job_data:
                        job_data[(source, model)] = pd.DataFrame({'job': [], 'decade': [decade], 'normalized_she': [np.nan]})
                    else:
                        extra_row = pd.DataFrame({'job': [np.nan], 'decade': [decade], 'normalized_she': [np.nan]})
                        job_data[(source, model)] = pd.concat([job_data[(source, model)], extra_row], ignore_index=True)

    # Ensure data is sorted by decade
    for key in job_data:
        job_data[key] = job_data[key].sort_values(by='decade')

    return job_data



def visualize_job_normalized_data(job_data, data_sources, model_types):
    """Visualize the ensemble normalized 'she' probabilities grouped by job across data sources with correlation."""
    unique_jobs = set()
    for data in job_data.values():
        unique_jobs.update(data['job'].dropna().unique())

    ensemble_averages = {source: {} for source in data_sources}

    for job in unique_jobs:
        # Calculate the ensemble average for each job in each data source
        for source in data_sources:
            ensemble_averages[source][job] = []
            for decade in range(1900, 2011, 10):
                decade_averages = []
                for model in model_types:
                    model_data = job_data.get((source, model))
                    if model_data is not None:
                        job_decade_data = model_data[(model_data['job'] == job) & (model_data['decade'] == decade)]['normalized_she']
                        decade_averages.extend(job_decade_data.values)
                # Calculate the mean of the normalized_she for this job and decade across all models
                ensemble_averages[source][job].append(np.nanmean(decade_averages))
        
    # Plot the ensemble averages for each job and calculate correlations
    for job in unique_jobs:
        plt.figure(figsize=(12, 6))
        data_lines = []
        for idx, source in enumerate(data_sources):
            if job in ensemble_averages[source]:
                line, = plt.plot(range(1900, 2011, 10), ensemble_averages[source][job], marker='o', label=f'Ensemble - {source}')
                data_lines.append(line.get_ydata())
        
        if len(data_lines) == 2:  # Check if we have two lines to compare
            correlation_matrix = np.corrcoef(data_lines[0], data_lines[1])
            correlation = correlation_matrix[0, 1]  # Get the correlation value
            plt.title(f'Ensemble Normalized "She" Probabilities for {job}\nCorrelation: {correlation:.2f}')
        else:
            plt.title(f'Ensemble Normalized "She" Probabilities for {job}')
        
        plt.xlabel('Decade')
        plt.ylabel('Ensemble Normalized Probability of "She"')
        plt.legend()
        plt.grid(True)
        st.pyplot()




def load_aggregated_data(data_source, model_name, base_path="results"):
    """Load aggregated P0 data for each decade."""
    data = {}
    for decade in range(1900, 2011, 10):
        file_path = os.path.join(base_path, data_source, model_name, 'aggregated_results', f"aggregated_{decade}_p0.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if not df.empty and 'P0_he' in df.columns and 'P0_she' in df.columns:
                df[['P0_he', 'P0_she']] = df[['P0_he', 'P0_she']].apply(pd.to_numeric, errors='coerce')
                df.dropna(subset=['P0_he', 'P0_she'], inplace=True)
                means = df[['P0_he', 'P0_she']].mean()
                ratio = means['P0_she'] / (means['P0_she'] + means['P0_he'])
                data[decade] = ratio
    return data

def load_aggregated_data_multiple(data_sources, model_types, base_path="results"):
    """Load aggregated P0 data for each decade for multiple data sources and model types."""
    data = {source: {model: {} for model in model_types} for source in data_sources}
    for source in data_sources:
        for model in model_types:
            for decade in range(1900, 2011, 10):
                file_path = os.path.join(base_path, source, model, 'aggregated_results', f"aggregated_{decade}_p0.csv")
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    if not df.empty and 'P0_he' in df.columns and 'P0_she' in df.columns:
                        df[['P0_he', 'P0_she']] = df[['P0_he', 'P0_she']].apply(pd.to_numeric, errors='coerce')
                        df.dropna(subset=['P0_he', 'P0_she'], inplace=True)
                        means = df[['P0_he', 'P0_she']].mean()
                        ratio = means['P0_she'] / (means['P0_she'] + means['P0_he'])
                        data[source][model][decade] = ratio
    return data


def plot_p0_trends(data):
    """Plot the average P0_she ratio over decades."""
    if not data:
        st.write("No data available.")
        return
    decades = list(data.keys())
    ratios = [data[decade] for decade in decades]
    plt.figure(figsize=(10, 5))
    plt.plot(decades, ratios, marker='o', color='magenta', label='P0_she/(P0_she + P0_he)')
    plt.title('Ratio of P0_she to Total P0 Values Over Decades')
    plt.xlabel('Decade')
    plt.ylabel('Ratio P0_she/(P0_she + P0_he)')
    plt.grid(True)
    plt.ylim(0, 1)
    plt.legend()
    st.pyplot(plt)

def plot_p0_trends_multiple(aggregated_data, data_sources, model_types):
    """Plot the average P0_she ratio over decades for multiple data sources and model types."""
    plt.figure(figsize=(10, 5))
    for source_idx, source in enumerate(data_sources):
        for model_idx, model in enumerate(model_types):
            if model in aggregated_data[source]:
                decades = sorted(aggregated_data[source][model].keys())
                ratios = [aggregated_data[source][model][decade] for decade in decades]
                plt.plot(decades, ratios, marker='o', label=f'{source} - {model}')
    plt.title('Ratio of P0_she to Total P0 Values Over Decades')
    plt.xlabel('Decade')
    plt.ylabel('P0_she/(P0_she + P0_he)')
    plt.grid(True)
    plt.legend()
    st.pyplot()


from itertools import combinations

def plot_normalized_she_trends(normalized_data, ensemble_data, model_types, data_sources):
    """Plot the average normalized_she values over decades for all model types and the ensemble for each data source."""
    colors = ['blue', 'green', 'red', 'magenta', 'cyan', 'orange']  # More colors for additional lines

    for source in data_sources:
        plt.figure(figsize=(10, 5))
        min_val, max_val = 1, 0  # Initialize min and max values for dynamic y-axis scaling
        model_data_lists = {}

        for model in model_types:
            model_data = [normalized_data[source][model].get(decade, np.nan) for decade in range(1900, 2011, 10)]
            model_data_lists[model] = model_data
            plt.plot(range(1900, 2011, 10), model_data, marker='o', label=f'{model}')

        # Calculate and display correlations for all pairs
        correlations = []
        for (model1, data1), (model2, data2) in combinations(model_data_lists.items(), 2):
            clean_data = [(x, y) for x, y in zip(data1, data2) if not (np.isnan(x) or np.isnan(y))]
            if clean_data:  # Proceed if there are pairs to compare
                x_values, y_values = zip(*clean_data)
                correlation_matrix = np.corrcoef(x_values, y_values)
                correlation = correlation_matrix[0, 1]  # Get the correlation value
                correlations.append(f'Correlation {model1}/{model2}: {correlation:.2f}')

        # Plot ensemble average
        ensemble_averages = [ensemble_data[source].get(decade, np.nan) for decade in range(1900, 2011, 10)]
        plt.plot(range(1900, 2011, 10), ensemble_averages, marker='o', linestyle='--', color='black', label='Ensemble')

        # Set plot title and display correlations
        correlation_text = "\n".join(correlations)
        plt.title(f'Normalized She Trend for {source}\n{correlation_text}')
        plt.xlabel('Decade')
        plt.ylabel('Average Normalized She')
        plt.grid(True)
        plt.legend()
        st.pyplot()




def plot_ensemble_comparison(ensemble_data, data_sources):
    plt.figure(figsize=(10, 5))
    ensemble_values = []
    for source in data_sources:
        ensemble_averages = [ensemble_data[source].get(decade, np.nan) for decade in range(1900, 2011, 10)]
        ensemble_values.append(ensemble_averages)
        plt.plot(range(1900, 2011, 10), ensemble_averages, marker='o', label=f'Ensemble - {source}')

    # Calculate correlation if both data sources have values
    if len(ensemble_values) == 2:
        # Flatten the list and remove pairs where at least one is NaN
        clean_values = [(x, y) for x, y in zip(*ensemble_values) if not (np.isnan(x) or np.isnan(y))]
        if clean_values:  # Proceed if there are pairs to compare
            x_values, y_values = zip(*clean_values)
            correlation_matrix = np.corrcoef(x_values, y_values)
            correlation = correlation_matrix[0, 1]  # Get the correlation value
            plt.title(f'Ensemble Comparison Across Data Sources\nCorrelation: {correlation:.2f}')
        else:
            plt.title('Ensemble Comparison Across Data Sources\nNo overlap for correlation')
    else:
        plt.title('Ensemble Comparison Across Data Sources')

    plt.xlabel('Decade')
    plt.ylabel('Average Normalized She')
    plt.grid(True)
    plt.legend()
    st.pyplot()




def main():
    st.set_page_config(page_title="Data Visualizations", layout="wide")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title('Data Visualizations')

    data_sources = ['case_law', 'ny_times']  # List of all data sources
    model_types = ['bert-base-uncased', 'distilbert-base-uncased', 'albert-base-v2']
    graph_options = ["P0_she Ratio Trend", "Normalized She Trend", "Jobs She Trend", "Ensemble Comparison"]

    selected_data_sources = st.sidebar.multiselect("Select Data Sources", data_sources, default=data_sources)
    selected_model_types = st.sidebar.multiselect("Select Model Types", model_types, default=model_types)
    selected_graphs = st.sidebar.radio("Select Graphs to Display", graph_options)

    if "P0_she Ratio Trend" == selected_graphs:
        aggregated_data_multiple = load_aggregated_data_multiple(selected_data_sources, selected_model_types)
        st.write("### P0_she Ratio Trend")
        plot_p0_trends_multiple(aggregated_data_multiple, selected_data_sources, selected_model_types)


    if "Jobs She Trend" == selected_graphs:
        job_data = load_job_normalized_data(selected_data_sources, selected_model_types)
        visualize_job_normalized_data(job_data, selected_data_sources, selected_model_types)


    if "Normalized She Trend" == selected_graphs:
        normalized_data, ensemble_data = load_normalized_data(selected_model_types, selected_data_sources)
        st.write("### Normalized She Trend")
        plot_normalized_she_trends(normalized_data, ensemble_data, selected_model_types, selected_data_sources)

    if "Ensemble Comparison" == selected_graphs:
        normalized_data, ensemble_data = load_normalized_data(selected_model_types, selected_data_sources)
        st.write("### Ensemble Comparison")
        plot_ensemble_comparison(ensemble_data, selected_data_sources)


if __name__ == "__main__":
    main()

