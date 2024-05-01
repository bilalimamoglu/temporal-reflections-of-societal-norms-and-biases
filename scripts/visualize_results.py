import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import json
import numpy as np
from itertools import combinations
from sklearn.metrics import mean_squared_error
from sklearn.cross_decomposition import CCA

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
    normalized_data = {source: {model: {} for model in model_types} for source in data_sources}
    ensemble_data = {source: {} for source in data_sources}
    base_results = {source: {model: None for model in model_types} for source in data_sources}

    for source in data_sources:
        for model in model_types:
            # Load base model results for horizontal line plotting
            base_file_path = os.path.join(base_path, source, model, 'base_results', 'p0', 'base_results.csv')
            if os.path.exists(base_file_path):
                base_df = pd.read_csv(base_file_path)
                if 'probabilities_tgt' in base_df.columns:
                    base_df['probabilities_tgt'] = base_df['probabilities_tgt'].apply(safe_json_loads)
                    base_df['tgt_he'] = base_df['probabilities_tgt'].apply(lambda x: x.get('he', 0.0001))
                    base_df['tgt_she'] = base_df['probabilities_tgt'].apply(lambda x: x.get('she', 0.0001))
                    total = base_df['tgt_he'] + base_df['tgt_she']
                    base_df['normalized_she'] = base_df['tgt_she'] / total
                    base_df['normalized_she'].fillna(0, inplace=True)
                    base_results[source][model] = base_df['normalized_she'].mean()

            for decade in range(1900, 2011, 10):
                decade_data = []
                file_path = os.path.join(base_path, source, model, 'raw_results', 'p0', f"{decade}_results_run_{1}_p0.csv")
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

        for decade in range(1900, 2011, 10):
            ensemble_data[source][decade] = np.mean([normalized_data[source][model][decade] for model in model_types])

    return normalized_data, ensemble_data, base_results



def visualize_job_normalized_data_with_occupation(job_data, data_sources, model_types, occupation_data):
    # Load one sample dataset to extract unique jobs
    sample_path = os.path.join('results', data_sources[0], model_types[0], 'raw_results', 'p0', '1900_results_run_1_p0.csv')
    sample_df = pd.read_csv(sample_path)
    unique_jobs = set(sample_df['job'].dropna().unique())  # Assuming 'job' column exists and names are standardized

    # Filter occupation data for those jobs only
    occupation_data = occupation_data[occupation_data['Occupation'].isin(unique_jobs)]
    ensemble_averages = {source: {} for source in data_sources}

    # For each job, plot the normalized data and compare with occupation data
    for job in unique_jobs:
        job_fig, job_ax = plt.subplots(figsize=(12, 6))
        all_ensemble_averages = []

        for source in data_sources:
            ensemble_averages[source][job] = []
            for decade in range(1900, 2011, 10):
                decade_averages = []
                for model in model_types:
                    model_data = job_data.get((source, model))
                    if model_data is not None:
                        job_decade_data = model_data[(model_data['job'] == job) & (model_data['decade'] == decade)]['normalized_she']
                        decade_averages.extend(job_decade_data.values)
                if decade_averages:
                    ensemble_average = np.nanmean(decade_averages)
                    ensemble_averages[source][job].append(ensemble_average)
                else:
                    ensemble_averages[source][job].append(np.nan)

            job_ax.plot(range(1900, 2011, 10), ensemble_averages[source][job], marker='o', label=f'Ensemble - {source}')
            all_ensemble_averages.append(ensemble_averages[source][job])

        # Calculate average ensemble across all data sources
        avg_ensemble = np.nanmean(all_ensemble_averages, axis=0)
        job_ax.plot(range(1900, 2011, 10), avg_ensemble, marker='o', linestyle='--', color='black', label='Average Ensemble')


        occupation_subset = occupation_data[occupation_data['Occupation'] == job]
        occupation_decades = occupation_subset['Decade'].values
        female_percentages = occupation_subset['Female'].values
        # Aligning occupation data with ensemble data decades for correlation calculation
        aligned_female_percentages = [female_percentages[occupation_decades == decade] for decade in range(1900, 2011, 10)]
        aligned_female_percentages = np.array([np.nanmean(values) if len(values) > 0 else np.nan for values in aligned_female_percentages])

        # Plot the aligned data for the occupation
        job_ax.plot(range(1900, 2011, 10), aligned_female_percentages, marker='s', linestyle='-', color='cyan', label=f'{job} - Occupation Data')

        # Calculating correlations
        correlation_texts = []

        # Aligning occupation data with ensemble data decades for correlation calculation
        aligned_female_percentages = np.array([np.nanmean(female_percentages[occupation_decades == decade]) if np.any(occupation_decades == decade) else np.nan for decade in range(1900, 2011, 10)])

        # Iterate over each data source for individual correlations
        for source in data_sources:
            source_data = np.array(ensemble_averages[source][job])  # Ensure it's an array for operations
            valid_indices = (~np.isnan(source_data) & ~np.isnan(aligned_female_percentages))

            if np.any(valid_indices):  # Ensure there are indices to compare
                source_correlation = np.corrcoef(source_data[valid_indices], aligned_female_percentages[valid_indices])[0, 1]
                correlation_texts.append(f'Correlation with {source}: {source_correlation:.2f}')
            else:
                correlation_texts.append(f'Correlation with {source}: Data mismatch')

        # Calculating correlation with the average ensemble
        avg_ensemble = np.nanmean(all_ensemble_averages, axis=0)
        valid_indices_avg = (~np.isnan(avg_ensemble) & ~np.isnan(aligned_female_percentages))

        if np.any(valid_indices_avg):
            avg_ensemble_correlation = np.corrcoef(avg_ensemble[valid_indices_avg], aligned_female_percentages[valid_indices_avg])[0, 1]
            correlation_texts.append(f'Correlation with Average Ensemble: {avg_ensemble_correlation:.2f}')
        else:
            correlation_texts.append('Correlation with Average Ensemble: Data mismatch')

        correlation_text = "\n".join(correlation_texts)
        job_ax.set_title(f'Ensemble Normalized "She" Probabilities for {job}\n{correlation_text}')
        job_ax.set_xlabel('Decade')
        job_ax.set_ylabel('Normalized Probability of "She"')
        job_ax.legend()
        job_ax.grid(True)
        st.pyplot(job_fig)



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
    unique_jobs = set()
    for data in job_data.values():
        unique_jobs.update(data['job'].dropna().unique())

    ensemble_averages = {source: {} for source in data_sources}

    for job in unique_jobs:
        job_fig, job_ax = plt.subplots(figsize=(12, 6))
        all_ensemble_averages = []

        for source in data_sources:
            ensemble_averages[source][job] = []
            for decade in range(1900, 2011, 10):
                decade_averages = []
                for model in model_types:
                    model_data = job_data.get((source, model))
                    if model_data is not None:
                        job_decade_data = model_data[(model_data['job'] == job) & (model_data['decade'] == decade)]['normalized_she']
                        decade_averages.extend(job_decade_data.values)
                ensemble_average = np.nanmean(decade_averages)
                ensemble_averages[source][job].append(ensemble_average)
            ensemble_line, = job_ax.plot(range(1900, 2011, 10), ensemble_averages[source][job], marker='o', label=f'Ensemble - {source}')
            all_ensemble_averages.append(ensemble_line.get_ydata())

        # Check if we have at least two ensembles to compare
        if len(all_ensemble_averages) > 1:
            clean_values = [list(filter(lambda x: not np.isnan(x), ensemble)) for ensemble in all_ensemble_averages]
            # Only compute correlation if there are no NaNs
            if clean_values and all(clean_values):
                correlation_matrix = np.corrcoef(clean_values)
                correlation = correlation_matrix[0, 1]  # Get the correlation value
                correlation_text = f'\nCorrelation: {correlation:.2f}'
            else:
                correlation_text = '\nInsufficient data for correlation'
        else:
            correlation_text = ''

        avg_ensemble = np.nanmean(all_ensemble_averages, axis=0)
        job_ax.plot(range(1900, 2011, 10), avg_ensemble, marker='o', linestyle='--', color='purple', label='Average Ensemble')

        job_ax.set_title(f'Ensemble Normalized "She" Probabilities for {job}{correlation_text}')
        job_ax.set_xlabel('Decade')
        job_ax.set_ylabel('Ensemble Normalized Probability of "She"')
        job_ax.legend()
        job_ax.grid(True)
        st.pyplot(job_fig)






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




def plot_normalized_she_trends(normalized_data, ensemble_data, base_results, model_types, data_sources):
    """Plot the average normalized_she values over decades for all model types and the ensemble for each data source."""
    colors = ['blue', 'green', 'red', 'magenta', 'cyan', 'orange']  # More colors for additional lines
    decades = np.array(range(1900, 2011, 10))

    for source in data_sources:
        plt.figure(figsize=(10, 5))
        min_val, max_val = 1, 0  # Initialize min and max values for dynamic y-axis scaling
        model_data_lists = {}

        for model in model_types:
            model_data = np.array([normalized_data[source][model].get(decade, np.nan) for decade in decades])
            model_data_lists[model] = model_data
            plt.plot(decades, model_data, marker='o', label=f'{model}', color=colors[model_types.index(model)])

            # Plot the base model results as a horizontal line
            if base_results[source][model] is not None:
                plt.axhline(y=base_results[source][model], color=colors[model_types.index(model)], linestyle='--', label=f'{model} Base')

        # Calculate and display Canonical Correlation Analysis for all pairs
        correlations = []
        for (model1, data1), (model2, data2) in combinations(model_data_lists.items(), 2):
            valid_indices = ~np.isnan(data1) & ~np.isnan(data2)
            if valid_indices.any():  # Proceed if there are non-NaN pairs to compare
                X = data1[valid_indices].reshape(-1, 1)
                Y = data2[valid_indices].reshape(-1, 1)
                cca = CCA(n_components=1)
                cca.fit(X, Y)
                X_c, Y_c = cca.transform(X, Y)
                correlation = np.corrcoef(X_c.T, Y_c.T)[0, 1]
                correlations.append(f'Correlation {model1}/{model2}: {correlation:.2f}')

        # Plot ensemble average
        ensemble_averages = np.array([ensemble_data[source].get(decade, np.nan) for decade in decades])
        valid_indices = ~np.isnan(ensemble_averages)
        plt.plot(decades[valid_indices], ensemble_averages[valid_indices], marker='o', linestyle='--', color='black', label='Ensemble')

        # Set plot title and display correlations
        correlation_text = "\n".join(correlations)
        plt.title(f'Normalized She Trend for {source}\n{correlation_text}')
        plt.xlabel('Decade')
        plt.ylabel('Average Normalized She')
        plt.grid(True)
        plt.legend()
        st.pyplot()



def compute_cohens_d(mean1, mean2, std1, std2, n1, n2):
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    cohens_d = (mean1 - mean2) / pooled_std
    return cohens_d

def get_cohens_d_for_decade(source, model_types, decade, base_path):
    he_means, she_means, he_stds, she_stds, n_he, n_she = [], [], [], [], [], []

    for model in model_types:
        file_path = os.path.join(base_path, source, model, 'aggregated_results', f"aggregated_{decade}_p0.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if 'P0_he' in df.columns and 'P0_she' in df.columns:
                he_values = np.log(df['P0_he'].replace(0, np.nan).dropna() + 0.0001)
                she_values = np.log(df['P0_she'].replace(0, np.nan).dropna() + 0.0001)

                if not he_values.empty and not she_values.empty:
                    he_means.append(he_values.mean())
                    she_means.append(she_values.mean())
                    he_stds.append(he_values.std())
                    she_stds.append(she_values.std())
                    n_he.append(len(he_values))
                    n_she.append(len(she_values))

    if he_means and she_means:
        overall_he_mean = np.nanmean(he_means)
        overall_she_mean = np.nanmean(she_means)
        overall_he_std = np.nanmean(he_stds)
        overall_she_std = np.nanmean(she_stds)
        total_n_he = sum(n_he)
        total_n_she = sum(n_she)
        cohens_d = compute_cohens_d(overall_he_mean, overall_she_mean, overall_he_std, overall_she_std, total_n_he, total_n_she)
        return cohens_d
    return np.nan

def plot_cohens_d_trends(data_sources, model_types, base_path="results"):
    decades = np.array(range(1900, 2011, 10))

    # Plotting across data sources with correlation and average line
    plt.figure(figsize=(12, 6))
    all_data_sources_cohens_d = []

    for source in data_sources:
        cohens_ds = [get_cohens_d_for_decade(source, model_types, decade, base_path) for decade in decades]
        all_data_sources_cohens_d.append(cohens_ds)
        plt.plot(decades, cohens_ds, marker='o', label=f'Cohen’s d for {source}')

    # Calculate and plot average Cohen's d across all data sources
    all_data_sources_cohens_d = np.array(all_data_sources_cohens_d, dtype=np.float64)
    valid_mask = ~np.isnan(all_data_sources_cohens_d).any(axis=0)
    average_cohens_d = np.nanmean(all_data_sources_cohens_d[:, valid_mask], axis=0)
    plt.plot(decades[valid_mask], average_cohens_d, marker='o', linestyle='--', color='black', label='Average Cohen’s d')

    # Calculate and display correlation between data sources if there are exactly two
    if len(data_sources) == 2:
        correlations = np.corrcoef(all_data_sources_cohens_d[0, valid_mask], all_data_sources_cohens_d[1, valid_mask])[0, 1]
        plt.title(f'Cohen’s d Across Data Sources\nCorrelation: {correlations:.2f}')
    else:
        plt.title('Cohen’s d Across Data Sources')

    plt.xlabel('Decade')
    plt.ylabel("Cohen's d (Gender Bias Magnitude)")
    plt.legend()
    plt.grid(True)
    st.pyplot()

def calculate_correlations_with_occupation(job_data, data_sources, model_types):
    # Load sample data to extract unique jobs
    occupation_data = pd.read_csv('data/occupation_decade_percentages_gender.csv')
    sample_path = os.path.join('results', data_sources[0], model_types[0], 'raw_results', 'p0', '1900_results_run_1_p0.csv')
    sample_df = pd.read_csv(sample_path)
    unique_jobs = set(sample_df['job'].dropna().unique())

    # Filter occupation data to match those jobs
    occupation_data = occupation_data[occupation_data['Occupation'].isin(unique_jobs)]
    correlation_results = []

    for job in unique_jobs:
        ensemble_averages = {source: [] for source in data_sources}

        # Process data for each decade and each source
        for decade in range(1900, 2011, 10):
            for source in data_sources:
                decade_averages = []
                for model in model_types:
                    model_data = job_data.get((source, model))
                    if model_data is not None:
                        job_decade_data = model_data[(model_data['job'] == job) & (model_data['decade'] == decade)]['normalized_she']
                        decade_averages.extend(job_decade_data.values)
                ensemble_averages[source].append(np.nanmean(decade_averages) if decade_averages else np.nan)

        # Calculate average ensemble across all data sources
        avg_ensemble = np.array([np.nanmean([ensemble_averages[source][i] for source in data_sources]) for i in range(len(range(1900, 2011, 10)))])

        # Occupation data alignment
        occupation_subset = occupation_data[occupation_data['Occupation'] == job]
        occupation_decades = occupation_subset['Decade'].values
        female_percentages = occupation_subset['Female'].values
        aligned_female_percentages = np.array([female_percentages[np.where(occupation_decades == decade)[0]].mean() if np.any(occupation_decades == decade) else np.nan for decade in range(1900, 2011, 10)])

        # Correlation calculations
        correlations = {}
        valid_indices = ~np.isnan(avg_ensemble) & ~np.isnan(aligned_female_percentages)

        # Average ensemble correlation
        if valid_indices.any():
            avg_ensemble_correlation = np.corrcoef(avg_ensemble[valid_indices], aligned_female_percentages[valid_indices])[0, 1]
        else:
            avg_ensemble_correlation = None

        correlations['Average Ensemble'] = avg_ensemble_correlation

        # Source-specific correlations
        for source in data_sources:
            source_data = np.array(ensemble_averages[source])
            valid_indices_source = ~np.isnan(source_data) & ~np.isnan(aligned_female_percentages)
            if valid_indices_source.any():
                source_correlation = np.corrcoef(source_data[valid_indices_source], aligned_female_percentages[valid_indices_source])[0, 1]
            else:
                source_correlation = None
            correlations[source] = source_correlation

        correlation_results.append({
            'Job': job,
            'Correlation with Average Ensemble': correlations['Average Ensemble'],
            'Correlation with ny_times': correlations.get('ny_times'),
            'Correlation with case_law': correlations.get('case_law')
        })

    # Create DataFrame and sort by correlation with average ensemble
    correlation_df = pd.DataFrame(correlation_results)
    correlation_df = correlation_df.sort_values('Correlation with Average Ensemble', ascending=False)
    return correlation_df


def calculate_ensemble_correlations(job_data, data_sources, model_types):
    # Ensure only 'case_law' and 'ny_times' are compared
    if 'case_law' not in data_sources or 'ny_times' not in data_sources:
        return pd.DataFrame()  # Return empty dataframe if either source is missing

    unique_jobs = set()
    for data in job_data.values():
        unique_jobs.update(data['job'].dropna().unique())

    ensemble_averages = {source: {} for source in data_sources}
    correlation_results = []

    for job in unique_jobs:
        for source in data_sources:
            ensemble_averages[source][job] = []

            for decade in range(1900, 2011, 10):
                decade_averages = []
                for model in model_types:
                    model_data = job_data.get((source, model))
                    if model_data is not None:
                        job_decade_data = model_data[(model_data['job'] == job) & (model_data['decade'] == decade)]['normalized_she']
                        decade_averages.extend(job_decade_data.values)
                
                if decade_averages:
                    ensemble_averages[source][job].append(np.nanmean(decade_averages))
                else:
                    ensemble_averages[source][job].append(np.nan)

        # Extract ensemble data for both sources for this job
        case_law_data = np.array(ensemble_averages['case_law'][job])
        ny_times_data = np.array(ensemble_averages['ny_times'][job])
        valid_indices = ~np.isnan(case_law_data) & ~np.isnan(ny_times_data)

        if valid_indices.any():
            correlation = np.corrcoef(case_law_data[valid_indices], ny_times_data[valid_indices])[0, 1]
        else:
            correlation = None  # Use None for no data or invalid correlation due to insufficient data

        # Store results in a list for later conversion to DataFrame
        correlation_results.append({
            'Job': job,
            'Correlation between Case Law and NY Times': correlation
        })

    # Convert list to DataFrame
    correlation_df = pd.DataFrame(correlation_results)
    correlation_df = correlation_df.dropna().sort_values('Correlation between Case Law and NY Times', ascending=False)

    return correlation_df



def plot_ensemble_comparison(ensemble_data, data_sources):
    plt.figure(figsize=(10, 5))
    all_ensemble_averages = []
    decades = np.array(range(1900, 2011, 10))

    # Plot individual ensembles
    for source in data_sources:
        ensemble_averages = [ensemble_data[source].get(decade, np.nan) for decade in decades]
        all_ensemble_averages.append(ensemble_averages)
        plt.plot(decades, ensemble_averages, marker='o', label=f'Ensemble - {source}')

    # Ensure there are no NaN values in the data for CCA
    clean_data = np.array(all_ensemble_averages, dtype=float).T
    valid_indices = ~np.isnan(clean_data).any(axis=1)

    if clean_data.shape[0] > 1 and len(data_sources) == 2:  # Check there's enough data and exactly two data sources
        # Prepare data for CCA
        X = clean_data[valid_indices, 0].reshape(-1, 1)
        Y = clean_data[valid_indices, 1].reshape(-1, 1)

        # Perform Canonical Correlation Analysis
        cca = CCA(n_components=1)
        cca.fit(X, Y)
        X_c, Y_c = cca.transform(X, Y)

        # Calculate correlation of the transformed components
        correlation = np.corrcoef(X_c.T, Y_c.T)[0, 1]

        plt.title(f'Ensemble Comparison Across Data Sources\nCorrelation: {correlation:.2f}')
    else:
        plt.title('Ensemble Comparison Across Data Sources\nNot enough data for CCA')

    # Calculate the average of the ensembles and plot it if possible
    if np.any(valid_indices):
        avg_ensemble = np.nanmean(clean_data[valid_indices], axis=1)
        plt.plot(decades[valid_indices], avg_ensemble, marker='o', linestyle='--', color='black', label='Average Ensemble')

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
    graph_options = ["P0_she Ratio Trend", "Normalized She Trend", "Jobs She Trend", "Jobs She Trend vs Occupation", "Cohens d", "Ensemble Comparison", "Calculate Occupation Correlations"]

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


    if "Jobs She Trend vs Occupation" == selected_graphs:
        job_data = load_job_normalized_data(selected_data_sources, selected_model_types)
        occupation_data = pd.read_csv('data/occupation_decade_percentages_gender.csv')
        st.dataframe(occupation_data)
        visualize_job_normalized_data_with_occupation(job_data, selected_data_sources, selected_model_types, occupation_data)


    if "Cohens d" == selected_graphs:
        plot_cohens_d_trends(data_sources, model_types, base_path="results")

    if "Normalized She Trend" == selected_graphs:
        normalized_data, ensemble_data, base_results = load_normalized_data(selected_model_types, selected_data_sources)
        st.write("### Normalized She Trend")
        st.write(base_results)
        plot_normalized_she_trends(normalized_data, ensemble_data, base_results, selected_model_types, selected_data_sources)

    if "Ensemble Comparison" == selected_graphs:
        normalized_data, ensemble_data, base_results = load_normalized_data(selected_model_types, selected_data_sources)
        st.write("### Ensemble Comparison")
        plot_ensemble_comparison(ensemble_data, selected_data_sources)


    if "Calculate Occupation Correlations" == selected_graphs:
        job_data = load_job_normalized_data(selected_data_sources, selected_model_types)
        df_occupation_correlations = calculate_correlations_with_occupation(job_data, selected_data_sources, selected_model_types)
        st.write(df_occupation_correlations)
        df_ensemble_correlations = calculate_ensemble_correlations(job_data, selected_data_sources, selected_model_types)
        st.write(df_ensemble_correlations)
        


if __name__ == "__main__":
    main()

