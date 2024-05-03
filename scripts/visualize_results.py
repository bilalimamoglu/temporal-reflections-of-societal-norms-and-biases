import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import json
import numpy as np
from itertools import combinations
from sklearn.metrics import mean_squared_error
from sklearn.cross_decomposition import CCA
from scipy.stats import f_oneway, pearsonr

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
    P0_ensemble_data = {source: {} for source in data_sources}
    base_results = {source: {model: None for model in model_types} for source in data_sources}
    P0_normalized_data = {source: {model: {} for model in model_types} for source in data_sources}
    base_P0_results = {source: {model: None for model in model_types} for source in data_sources}

    log_data = {source: {model: {} for model in model_types} for source in data_sources}
    log_ensemble_data = {source: {} for source in data_sources}
    base_log_results = {source: {model: None for model in model_types} for source in data_sources}


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

                    base_df['P0_normalized_she'] = base_df['P0_she'] / (base_df['P0_he'] + base_df['P0_she'])
                    base_df['P0_normalized_she'].fillna(0, inplace=True)
                    base_df['log_prob_she'] = np.log(base_df['P0_she']) - np.log(base_df['P0_he'])

                    base_results[source][model] = base_df['normalized_she'].mean()
                    base_P0_results[source][model] = base_df['P0_normalized_she'].mean()
                    base_log_results[source][model] = base_results[source][model].mean()

            for decade in range(1900, 2011, 10):
                decade_data = []
                P0_decade_data = []
                log_decade_data = []
                for run in range(1, 4):  # Handling three runs
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

                            df['P0_normalized_she'] = df['P0_she'] / (df['P0_he'] + df['P0_she'])
                            df['P0_normalized_she'].fillna(0, inplace=True)
                            P0_decade_data.append(df['P0_normalized_she'].mean())

                            df['log_prob_she'] = np.log(df['P0_she']) - np.log(df['P0_he'])
                            
                            log_decade_data.append(df['log_prob_she'].mean())

                normalized_data[source][model][decade] = np.mean(decade_data) if decade_data else 0
                P0_normalized_data[source][model][decade] = np.mean(P0_decade_data) if P0_decade_data else 0
                log_data[source][model][decade] = np.mean(log_decade_data) if log_decade_data else 0

        for decade in range(1900, 2011, 10):
            ensemble_data[source][decade] = np.mean([normalized_data[source][model][decade] for model in model_types])
            P0_ensemble_data[source][decade] = np.mean([P0_normalized_data[source][model][decade] for model in model_types])
            log_ensemble_data[source][decade] = np.mean([log_data[source][model][decade] for model in model_types])

    return normalized_data, P0_normalized_data, ensemble_data, P0_ensemble_data, base_results, base_P0_results, log_data, log_ensemble_data, base_log_results

def calculate_log_probability_bias_score(model_types, data_sources, base_path="results"):
    bias_scores = {source: {model: {} for model in model_types} for source in data_sources}
    
    for source in data_sources:
        for model in model_types:
            for decade in range(1900, 2011, 10):
                decade_bias_scores = []
                for run in range(1, 4):  # Assuming three runs for averaging
                    file_path = os.path.join(base_path, source, model, 'raw_results', 'p0', f"{decade}_results_run_{run}_p0.csv")
                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path)
                        if 'P0_he' in df.columns and 'P0_she' in df.columns:
                            # Calculate log probabilities and then the bias score
                            log_prob_he = np.log(df['P0_he'] + 0.0001)  # Adding a small number to avoid log(0)
                            log_prob_she = np.log(df['P0_she'] + 0.0001)
                            log_prob_bias = np.mean(log_prob_she) - np.mean(log_prob_he)
                            decade_bias_scores.append(log_prob_bias)
                # Average the bias scores over runs if available
                if decade_bias_scores:
                    bias_scores[source][model][decade] = np.mean(decade_bias_scores)
                else:
                    bias_scores[source][model][decade] = np.nan

    # Plotting the results
    plt.figure(figsize=(10, 6))
    for model_index, model in enumerate(model_types):
        decades = list(range(1900, 2011, 10))
        model_scores = [bias_scores[data_sources[0]][model][dec] for dec in decades]  # Example uses first data source
        plt.plot(decades, model_scores, label=model, marker='o')

    plt.title('Log Probability Bias Score Over Decades')
    plt.xlabel('Decade')
    plt.ylabel('Log Probability Bias Score (she - he)')
    plt.legend()
    plt.grid(True)
    st.pyplot()

from scipy.stats import linregress
import seaborn as sns


def visualize_job_normalized_data_with_occupation(job_data, data_sources, model_types, occupation_data, data_field):
    # Load one sample dataset to extract unique jobs
    sample_path = os.path.join('results', data_sources[0], model_types[0], 'raw_results', 'p0', '1900_results_run_1_p0.csv')
    sample_df = pd.read_csv(sample_path)
    unique_jobs = set(sample_df['job'].dropna().unique())

    # Filter occupation data for those jobs only
    occupation_data = occupation_data[occupation_data['Occupation'].isin(unique_jobs)]

    sns.set(style="darkgrid")  
    palette = sns.color_palette("bright")  

    for job in unique_jobs:
        job_fig, job_ax = plt.subplots(figsize=(12, 6))
        all_ensemble_averages = []
        all_occupation_values = []

        decades = np.array(range(1900, 2011, 10))
        for decade in decades:
            decade_averages = []
            for source in data_sources:
                for model in model_types:
                    model_data = job_data.get((source, model))
                    if model_data is not None:
                        job_decade_data = model_data[(model_data['job'] == job) & (model_data['decade'] == decade)][data_field]
                        decade_averages.extend(job_decade_data.dropna().values)
            all_ensemble_averages.append(np.nanmean(decade_averages) if decade_averages else np.nan)

            decade_occupation_data = occupation_data[(occupation_data['Occupation'] == job) & (occupation_data['Decade'] == decade)]['Female']
            all_occupation_values.append(decade_occupation_data.mean() if not decade_occupation_data.empty else np.nan)

        if np.all(np.isnan(all_ensemble_averages)) or np.all(np.isnan(all_occupation_values)):
            plt.close(job_fig)  # Close the figure and skip this job if data is insufficient
            continue

        # Plotting the data
        sns.lineplot(x=decades, y=all_ensemble_averages, marker='o', linestyle='-', color=palette[0], label='Average Ensemble', ax=job_ax)

        # Fit and plot linear trend line for average ensemble
        valid_indices_ensemble = ~np.isnan(all_ensemble_averages)
        if valid_indices_ensemble.any():
            slope, intercept, r_value, p_value, std_err = linregress(decades[valid_indices_ensemble], np.array(all_ensemble_averages)[valid_indices_ensemble])
            sns.lineplot(x=decades, y=intercept + slope * decades, color=palette[1], linestyle='--', label=f'Trend Line for Average Ensemble: R^2={r_value**2:.2f}', ax=job_ax)

        # Set secondary Y-axis for relative occupation percentages
        sec_ax = job_ax.twinx()
        sns.lineplot(x=decades, y=all_occupation_values, marker='D', linestyle='-', color=palette[2], label='Relative Occupation Percentage', ax=sec_ax)
        
        # Fit and plot linear trend line for occupation data
        valid_indices_occupation = ~np.isnan(all_occupation_values)
        if valid_indices_occupation.any():
            slope, intercept, r_value, p_value, std_err = linregress(decades[valid_indices_occupation], np.array(all_occupation_values)[valid_indices_occupation])
            sns.lineplot(x=decades, y=intercept + slope * decades, color=palette[3], linestyle='--', label=f'Trend Line for Occupation Data: R^2={r_value**2:.2f}', ax=sec_ax)

        # Display correlations
        valid_indices = valid_indices_ensemble & valid_indices_occupation
        if valid_indices.any():
            pearson_corr, pearson_p_value = pearsonr(np.array(all_ensemble_averages)[valid_indices], np.array(all_occupation_values)[valid_indices])
            spearman_corr, spearman_p_value = spearmanr(np.array(all_ensemble_averages)[valid_indices], np.array(all_occupation_values)[valid_indices])
            correlation_text = (f'Pearson Correlation: {pearson_corr:.2f} (p-value: {pearson_p_value:.3f}), '
                                f'Spearman Correlation: {spearman_corr:.2f} (p-value: {spearman_p_value:.3f})')
            job_ax.set_title(f'Normalized "She" Probabilities and Occupation Data for {job}\n{correlation_text}')

        job_ax.set_xlabel('Decade')
        job_ax.set_ylabel('Normalized Probability of "She"')
        sec_ax.set_ylabel('Relative Occupation Percentage (%)', color='magenta')
        job_ax.legend(loc='upper left')
        sec_ax.legend(loc='upper right')
        job_ax.grid(True)
        st.pyplot(job_fig)



def anova_job_data_with_occupation(job_data, data_sources, model_types, occupation_data, data_field):
    results = []
    sample_path = os.path.join('results', data_sources[0], model_types[0], 'raw_results', 'p0', '1900_results_run_1_p0.csv')
    sample_df = pd.read_csv(sample_path)
    unique_jobs = set(sample_df['job'].dropna().unique())

    # Filter occupation data for those jobs only
    occupation_data = occupation_data[occupation_data['Occupation'].isin(unique_jobs)]

    for job in unique_jobs:
        data_for_anova = {source: [] for source in data_sources}
        data_for_anova['Occupation'] = []

        for decade in range(1900, 2011, 10):
            for source in data_sources:
                decade_averages = []
                for model in model_types:
                    model_data = job_data.get((source, model))
                    if model_data is not None:
                        job_decade_data = model_data[(model_data['job'] == job) & (model_data['decade'] == decade)][data_field]
                        decade_averages.extend(job_decade_data.values)
                if decade_averages:
                    data_for_anova[source].append(np.nanmean(decade_averages))

            # Add occupation data for this decade
            occupation_subset = occupation_data[(occupation_data['Occupation'] == job) & (occupation_data['Decade'] == decade)]
            if not occupation_subset.empty:
                data_for_anova['Occupation'].append(occupation_subset['Female'].mean())
            else:
                data_for_anova['Occupation'].append(np.nan)

        # Perform ANOVA across the collected data for each source plus the occupation data
        lists_to_compare = [np.array(data_for_anova[src]).astype(np.float64) for src in data_for_anova if any(~np.isnan(data_for_anova[src]))]
        if len(lists_to_compare) > 1:
            F_statistic, p_value = f_oneway(*lists_to_compare)
            results.append({
                'Job': job,
                'F-statistic': F_statistic,
                'p-value': p_value
            })

    return pd.DataFrame(results)


def load_job_normalized_data(data_sources, model_types, base_path="results"):
    """Aggregate normalized_she and P0_normalized_she data for each job across all decades, model types, and data sources."""
    job_data = {}
    for source in data_sources:
        for model in model_types:
            for decade in range(1900, 2011, 10):
                for run in range(1, 4):  # Handling three runs
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

                            total_p0 = df['P0_he'] + df['P0_she']
                            df['P0_normalized_she'] = df['P0_she'] / total_p0
                            df['P0_normalized_she'].fillna(0, inplace=True)

                            df['log_prob_she'] = np.log(df['P0_she']) - np.log(df['P0_he'])
                            
                            df['job'] = df['job']
                            df['decade'] = decade
                            key = (source, model)
                            if key not in job_data:
                                job_data[key] = df[['job', 'decade', 'normalized_she', 'P0_normalized_she', 'log_prob_she']]
                            else:
                                job_data[key] = pd.concat([job_data[key], df[['job', 'decade', 'normalized_she', 'P0_normalized_she', 'log_prob_she']]], ignore_index=True)
                    else:
                        # If no file is found, fill the decade with NaNs for this model and source
                        extra_row = pd.DataFrame({'job': [np.nan], 'decade': [decade], 'normalized_she': [np.nan], 'P0_normalized_she': [np.nan]})
                        key = (source, model)
                        if key not in job_data:
                            job_data[key] = extra_row
                        else:
                            job_data[key] = pd.concat([job_data[key], extra_row], ignore_index=True)

    # Ensure data is sorted by decade
    for key in job_data:
        job_data[key] = job_data[key].sort_values(by='decade')

    return job_data



def visualize_job_normalized_data(job_data, data_sources, model_types, data_field):
    """Visualize job normalized data, specifically focusing on correlation between case_law and ny_times."""
    unique_jobs = set()
    for data in job_data.values():
        unique_jobs.update(data['job'].dropna().unique())

    # Ensure that both case_law and ny_times are in data_sources
    if 'case_law' not in data_sources or 'ny_times' not in data_sources:
        print("Both case_law and ny_times must be present in data_sources.")
        return

    ensemble_averages = {source: {} for source in data_sources}

    for job in unique_jobs:
        job_fig, job_ax = plt.subplots(figsize=(12, 6))
        data_by_source = []

        for source in ['case_law', 'ny_times']:
            ensemble_averages[source][job] = []
            for decade in range(1900, 2011, 10):
                decade_averages = []
                for model in model_types:
                    model_data = job_data.get((source, model))
                    if model_data is not None:
                        job_decade_data = model_data[(model_data['job'] == job) & (model_data['decade'] == decade)][data_field]
                        decade_averages.extend(job_decade_data.dropna().values)  # Ensure only non-NaN values are extended
                if decade_averages:
                    ensemble_average = np.nanmean(decade_averages)
                    ensemble_averages[source][job].append(ensemble_average)
                else:
                    ensemble_averages[source][job].append(np.nan)
            data_by_source.append(ensemble_averages[source][job])
            job_ax.plot(range(1900, 2011, 10), ensemble_averages[source][job], marker='o', label=f'Ensemble - {source}')

        # Ensuring both sources have valid data for correlation
        if all(data_by_source):
            valid_indices = ~np.isnan(data_by_source[0]) & ~np.isnan(data_by_source[1])
            if valid_indices.any():
                # Perform Pearson and Spearman correlations on valid data only
                pearson_corr, pearson_p_value = pearsonr(np.array(data_by_source[0])[valid_indices], np.array(data_by_source[1])[valid_indices])
                spearman_corr, spearman_p_value = spearmanr(np.array(data_by_source[0])[valid_indices], np.array(data_by_source[1])[valid_indices])
                correlation_text = (f'\nPearson Correlation: {pearson_corr:.2f} (p-value: {pearson_p_value:.3f}), '
                                    f'Spearman Correlation: {spearman_corr:.2f} (p-value: {spearman_p_value:.3f})')
            else:
                correlation_text = '\nInsufficient data for correlation'
        else:
            correlation_text = '\nInsufficient data for correlation'

        job_ax.set_title(f'Ensemble {data_field} Probabilities for {job}{correlation_text}')
        job_ax.set_xlabel('Decade')
        job_ax.set_ylabel(f'{data_field}')
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




import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import pearsonr

def plot_normalized_she_trends(normalized_data, ensemble_data, base_results, model_types, data_sources):
    """Plot the average normalized_she values over decades for all model types and the ensemble for each data source."""
    colors = ['blue', 'green', 'red', 'magenta', 'cyan', 'orange']  # More colors for additional lines
    decades = np.array(range(1900, 2011, 10))

    for source in data_sources:
        plt.figure(figsize=(10, 5))
        model_data_lists = {}

        for model in model_types:
            model_data = np.array([normalized_data[source][model].get(decade, np.nan) for decade in decades])
            model_data_lists[model] = model_data
            plt.plot(decades, model_data, marker='o', label=f'{model}', color=colors[model_types.index(model)])

            # Plot the base model results as a horizontal line
            if base_results[source][model] is not None:
                plt.axhline(y=base_results[source][model], color=colors[model_types.index(model)], linestyle='--', label=f'{model} Base')

        # Plot ensemble average
        ensemble_averages = np.array([ensemble_data[source].get(decade, np.nan) for decade in decades])
        plt.plot(decades[~np.isnan(ensemble_averages)], ensemble_averages[~np.isnan(ensemble_averages)], marker='o', linestyle='--', color='black', label='Ensemble')

        # Calculate and display Pearson correlation for all pairs
        correlations = []
        for (model1, data1), (model2, data2) in combinations(model_data_lists.items(), 2):
            valid_indices = ~np.isnan(data1) & ~np.isnan(data2)
            if valid_indices.any():
                correlation, p_value = pearsonr(data1[valid_indices], data2[valid_indices])
                correlations.append(f'Correlation {model1}/{model2}: {correlation:.2f} (p={p_value:.4f})')

        # Set plot title and display correlations
        correlation_text = "\n".join(correlations)
        plt.title(f'Normalized She Trend for {source}\n{correlation_text}')
        plt.xlabel('Decade')
        plt.ylabel('Average Normalized She')
        plt.grid(True)
        plt.legend()
        st.pyplot()

def plot_P0_normalized_she_trends(normalized_data, ensemble_data, base_results, model_types, data_sources):
    """Plot the average normalized_she values over decades for all model types and the ensemble for each data source."""
    colors = ['blue', 'green', 'red', 'magenta', 'cyan', 'orange']  # More colors for additional lines
    decades = np.array(range(1900, 2011, 10))

    for source in data_sources:
        plt.figure(figsize=(10, 5))
        model_data_lists = {}

        for model in model_types:
            model_data = np.array([normalized_data[source][model].get(decade, np.nan) for decade in decades])
            model_data_lists[model] = model_data
            plt.plot(decades, model_data, marker='o', label=f'{model}', color=colors[model_types.index(model)])

            # Plot the base model results as a horizontal line
            if base_results[source][model] is not None:
                plt.axhline(y=base_results[source][model], color=colors[model_types.index(model)], linestyle='--', label=f'{model} Base')

        # Plot ensemble average
        ensemble_averages = np.array([ensemble_data[source].get(decade, np.nan) for decade in decades])
        plt.plot(decades[~np.isnan(ensemble_averages)], ensemble_averages[~np.isnan(ensemble_averages)], marker='o', linestyle='--', color='black', label='Ensemble')

        # Calculate and display Pearson and Spearman correlations for all pairs
        correlations = []
        for (model1, data1), (model2, data2) in combinations(model_data_lists.items(), 2):
            valid_indices = ~np.isnan(data1) & ~np.isnan(data2)
            if valid_indices.any():
                pearson_corr, pearson_p_value = pearsonr(data1[valid_indices], data2[valid_indices])
                spearman_corr, spearman_p_value = spearmanr(data1[valid_indices], data2[valid_indices])
                correlations.append(f'Pearson {model1}/{model2}: {pearson_corr:.2f} (p={pearson_p_value:.4f}), '
                                    f'Spearman {model1}/{model2}: {spearman_corr:.2f} (p={spearman_p_value:.4f})')

        # Set plot title and display correlations
        correlation_text = "\n".join(correlations)
        plt.title(f'Normalized She Trend for {source}\n{correlation_text}')
        plt.xlabel('Decade')
        plt.ylabel('Average Normalized She')
        plt.grid(True)
        plt.legend()
        st.pyplot()

import matplotlib.pyplot as plt
import numpy as np

def plot_P0_base_model(base_results, model_types, data_sources):
    """Plot the base model P0 normalized she values for each model type for the first data source only, with annotated bars."""

    if not data_sources:
        print("No data sources available.")
        return
    
    source = data_sources[0]  # Consider only the first data source
    
    # Gather values for each model type from the first data source
    values = [base_results[source][model] if model in base_results[source] else 0 for model in model_types]
    
    # Colors for each bar
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_types)))
    
    # Create a figure and a bar plot
    plt.figure(figsize=(8, 4))  # Smaller figure size for stylish look
    bars = plt.bar(model_types, values, color=colors, alpha=0.8)
    
    # Add text annotation inside bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}',
                 ha='center', va='bottom', color='black', fontweight='bold')
    
    plt.title(f'Base Model Normalized She Results in {source}')
    plt.xlabel('Model Type')
    plt.ylabel('Average Normalized She')
    plt.ylim(0, max(values) + 0.1)  # Set y-limit to make space for text
    plt.grid(True, linestyle='--', alpha=0.5)
    st.pyplot()


def get_cohens_d_for_decade(source, model_types, decade):
    he_means, she_means = [], []
    for model in model_types:
        model_he_means, model_she_means = [], []
        for run in range(1, 4):  # Assuming there are 3 runs
            file_path = os.path.join('results', source, model, 'raw_results', 'p0', f"{decade}_results_run_{run}_p0.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                if 'P0_he' in df.columns and 'P0_she' in df.columns:
                    model_he_means.append(df['P0_he'].mean())
                    model_she_means.append(df['P0_she'].mean())

        if model_he_means and model_she_means:
            he_means.extend(model_he_means)
            she_means.extend(model_she_means)

    if he_means and she_means:
        cohens_d = compute_cohens_d(np.mean(he_means), np.mean(she_means), np.std(he_means, ddof=1), np.std(she_means, ddof=1), len(he_means), len(she_means))
        return cohens_d
    return np.nan

def compute_cohens_d(mean1, mean2, std1, std2, n1, n2):
    """ Compute Cohen's d for two sets of data. """
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    return (mean1 - mean2) / pooled_std

def plot_cohens_d_trends(data_sources, model_types):
    decades = np.array(range(1900, 2011, 10))

    # Plotting across data sources with correlation and average line
    plt.figure(figsize=(12, 6))
    all_data_sources_cohens_d = []

    for source in data_sources:
        cohens_ds = [get_cohens_d_for_decade(source, model_types, decade) for decade in decades]
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



def calculate_correlations_with_occupation(job_data, data_sources, model_types, data_field):
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
                        job_decade_data = model_data[(model_data['job'] == job) & (model_data['decade'] == decade)][data_field]
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

        # Average ensemble correlation - Pearson and Spearman
        if valid_indices.any():
            pearson_corr, pearson_p_value = pearsonr(avg_ensemble[valid_indices], aligned_female_percentages[valid_indices])
            spearman_corr, spearman_p_value = spearmanr(avg_ensemble[valid_indices], aligned_female_percentages[valid_indices])
            correlations['Average Ensemble Pearson'] = pearson_corr
            correlations['P-value Average Ensemble Pearson'] = pearson_p_value
            correlations['Average Ensemble Spearman'] = spearman_corr
            correlations['P-value Average Ensemble Spearman'] = spearman_p_value
        else:
            correlations['Average Ensemble Pearson'] = None
            correlations['P-value Average Ensemble Pearson'] = None
            correlations['Average Ensemble Spearman'] = None
            correlations['P-value Average Ensemble Spearman'] = None

        # Source-specific correlations - Pearson and Spearman
        for source in data_sources:
            source_data = np.array(ensemble_averages[source])
            valid_indices_source = ~np.isnan(source_data) & ~np.isnan(aligned_female_percentages)
            if valid_indices_source.any():
                pearson_corr, pearson_p_value = pearsonr(source_data[valid_indices_source], aligned_female_percentages[valid_indices_source])
                spearman_corr, spearman_p_value = spearmanr(source_data[valid_indices_source], aligned_female_percentages[valid_indices_source])
                correlations[f'Correlation with {source} Pearson'] = pearson_corr
                correlations[f'P-value with {source} Pearson'] = pearson_p_value
                correlations[f'Correlation with {source} Spearman'] = spearman_corr
                correlations[f'P-value with {source} Spearman'] = spearman_p_value
            else:
                correlations[f'Correlation with {source} Pearson'] = None
                correlations[f'P-value with {source} Pearson'] = None
                correlations[f'Correlation with {source} Spearman'] = None
                correlations[f'P-value with {source} Spearman'] = None

        correlation_results.append({
            'Job': job,
            **correlations
        })

    # Create DataFrame and sort by Pearson correlation with average ensemble
    correlation_df = pd.DataFrame(correlation_results)
    correlation_df = correlation_df.sort_values('Average Ensemble Pearson', ascending=False, na_position='last')
    
    return correlation_df



def calculate_ensemble_correlations(job_data, data_sources, model_types, data_field):
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
                        job_decade_data = model_data[(model_data['job'] == job) & (model_data['decade'] == decade)][data_field]
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
            # Pearson correlation and its p-value
            pearson_corr, pearson_p_value = pearsonr(case_law_data[valid_indices], ny_times_data[valid_indices])
            # Spearman correlation and its p-value
            spearman_corr, spearman_p_value = spearmanr(case_law_data[valid_indices], ny_times_data[valid_indices])
        else:
            pearson_corr, pearson_p_value, spearman_corr, spearman_p_value = None, None, None, None

        # Store results in a list for later conversion to DataFrame
        correlation_results.append({
            'Job': job,
            'Pearson Correlation between Case Law and NY Times': pearson_corr,
            'Pearson P-value': pearson_p_value,
            'Spearman Correlation between Case Law and NY Times': spearman_corr,
            'Spearman P-value': spearman_p_value
        })

    # Convert list to DataFrame
    correlation_df = pd.DataFrame(correlation_results)
    correlation_df = correlation_df.dropna().sort_values('Pearson Correlation between Case Law and NY Times', ascending=False)

    return correlation_df



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

def plot_ensemble_comparison(ensemble_data, data_sources):
    plt.figure(figsize=(10, 5))
    all_ensemble_averages = []
    decades = np.array(range(1900, 2011, 10))

    # Prepare to collect differences for correlation of changes
    all_differences = []

    # Plot individual ensembles
    for source in data_sources:
        ensemble_averages = [ensemble_data[source].get(decade, np.nan) for decade in decades]
        all_ensemble_averages.append(ensemble_averages)
        plt.plot(decades, ensemble_averages, marker='o', label=f'Ensemble - {source}')
        
        # Calculate differences for this source
        differences = np.diff(ensemble_averages)
        all_differences.append(differences)

    # Ensure there are no NaN values in the data
    clean_data = np.array(all_ensemble_averages, dtype=float).T
    valid_indices = ~np.isnan(clean_data).any(axis=1)

    if len(data_sources) == 2 and valid_indices.any():  # Ensure exactly two data sources and valid data
        # Extract data for both sources
        data1 = clean_data[valid_indices, 0]
        data2 = clean_data[valid_indices, 1]

        # Calculate Pearson correlation and p-value for absolute values
        correlation, p_value = pearsonr(data1, data2)
        # Calculate Spearman correlation and p-value for absolute values
        spearman_corr, spearman_p_value = spearmanr(data1, data2)

        # Calculate Cross-Correlation at zero lag
        cross_correlation = np.correlate(data1, data2)[0] / (len(data1) * data1.std() * data2.std())

        # Calculate Pearson correlation and p-value for changes
        if len(all_differences[0]) > 1 and len(all_differences[1]) > 1:  # Ensure there are enough points for changes
            changes_correlation, changes_p_value = pearsonr(all_differences[0], all_differences[1])
            changes_text = f', Changes Correlation: {changes_correlation:.2f}, Changes P-value: {changes_p_value:.3f}'
        else:
            changes_text = ', Insufficient data for Changes Correlation'
        
        plt.title(f'Ensemble Comparison Across Data Sources\nPearson Correlation: {correlation:.2f}, P-value: {p_value:.3f}\nSpearman Correlation: {spearman_corr:.2f}, Spearman P-value: {spearman_p_value:.3f}{changes_text}\nCross-Correlation (Zero lag): {cross_correlation:.2f}')
    else:
        plt.title('Ensemble Comparison Across Data Sources\nNot enough data for correlation')

    # Calculate the average of the ensembles and plot it
    if np.any(valid_indices):
        avg_ensemble = np.nanmean(clean_data[valid_indices], axis=1)
        plt.plot(decades[valid_indices], avg_ensemble, marker='o', linestyle='--', color='black', label='Average Ensemble')

    plt.xlabel('Decade')
    plt.ylabel('Average Normalized She')
    plt.grid(True)
    plt.legend()
    st.pyplot()

def compare_bias_scores(normalized_ensemble_data, log_ensemble_data, data_sources):
    plt.figure(figsize=(12, 6))
    decades = np.array(range(1900, 2011, 10))
    
    # Prepare to plot data
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Colors for the plots
    color_norm = 'tab:blue'
    color_log = 'tab:red'
    
    # Setting labels for axes
    ax1.set_xlabel('Decade')
    ax1.set_ylabel('Normalized Probability Bias Score', color=color_norm)
    ax1.tick_params(axis='y', labelcolor=color_norm)
    
    # Calculate and plot the average of normalized data across sources
    average_norm_data = np.nanmean([[normalized_ensemble_data[source].get(dec, np.nan) for dec in decades] for source in data_sources], axis=0)
    ax1.plot(decades, average_norm_data, marker='o', linestyle='-', label='Average Normalized', color=color_norm)

    # Creating a second y-axis for log probability bias score
    ax2 = ax1.twinx()
    ax2.set_ylabel('Log Probability Bias Score', color=color_log)
    ax2.tick_params(axis='y', labelcolor=color_log)

    # Calculate and plot the average of log data across sources
    average_log_data = np.nanmean([[log_ensemble_data[source].get(dec, np.nan) for dec in decades] for source in data_sources], axis=0)
    ax2.plot(decades, average_log_data, marker='x', linestyle='--', label='Average Log', color=color_log)

    # Check for valid indices to compute correlations
    valid_indices = ~np.isnan(average_norm_data) & ~np.isnan(average_log_data)
    if valid_indices.any():
        pearson_corr, pearson_p_value = pearsonr(average_norm_data[valid_indices], average_log_data[valid_indices])
        spearman_corr, spearman_p_value = spearmanr(average_norm_data[valid_indices], average_log_data[valid_indices])

        # Display correlation information
        plt.figtext(0.5, 0.01, f'Pearson Correlation: {pearson_corr:.2f} (p={pearson_p_value:.3f}), '
                                f'Spearman Correlation: {spearman_corr:.2f} (p={spearman_p_value:.3f})', 
                    ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

    # Set plot titles and legend
    fig.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05))
    plt.title('Comparison of Average Normalized and Log Probability Bias Scores Across Data Sources')
    plt.grid(True)
    st.pyplot()


def plot_P0_ensemble_comparison(ensemble_data, data_sources):
    plt.figure(figsize=(10, 5))
    decades = np.array(range(1900, 2011, 10))
    ensemble_dict = {source: [] for source in data_sources}
    ensemble_dict['Average Ensemble'] = []

    # Prepare to collect differences for correlation of changes
    differences_dict = {source: [] for source in data_sources}

    # Plot individual ensembles
    for source in data_sources:
        ensemble_averages = [ensemble_data[source].get(decade, np.nan) for decade in decades]
        ensemble_dict[source] = ensemble_averages
        plt.plot(decades, ensemble_averages, marker='o', label=f'Ensemble - {source}')
        
        # Calculate differences for changes correlation
        if len(ensemble_averages) > 1:
            differences_dict[source] = np.diff(ensemble_averages)

    # Compute and plot the average ensemble
    if len(data_sources) > 0:
        for i in range(len(decades)):
            avg_value = np.nanmean([ensemble_data[src].get(decades[i], np.nan) for src in data_sources])
            ensemble_dict['Average Ensemble'].append(avg_value)
        plt.plot(decades, ensemble_dict['Average Ensemble'], marker='o', linestyle='--', color='black', label='Average Ensemble')

    # Calculate Pearson correlation for the two main data sources if exactly two are specified
    if len(data_sources) == 2:
        data1 = np.array(ensemble_dict[data_sources[0]])
        data2 = np.array(ensemble_dict[data_sources[1]])
        valid_indices = ~np.isnan(data1) & ~np.isnan(data2)
        if valid_indices.any():
            correlation, p_value = pearsonr(data1[valid_indices], data2[valid_indices])
            # Calculate changes correlation
            changes_correlation, changes_p_value = pearsonr(
                differences_dict[data_sources[0]], differences_dict[data_sources[1]]
            ) if len(differences_dict[data_sources[0]]) > 0 and len(differences_dict[data_sources[1]]) > 0 else (None, None)
            plt.title(f'Ensemble Comparison Across Data Sources\nCorrelation: {correlation:.2f}, P-value: {p_value:.3f}, Changes Correlation: {changes_correlation:.2f}, Changes P-value: {changes_p_value:.3f}')
        else:
            plt.title('Ensemble Comparison Across Data Sources\nNot enough data for correlation')
    else:
        plt.title('Ensemble Comparison Across Data Sources')

    plt.xlabel('Decade')
    plt.ylabel('Average Normalized She')
    plt.grid(True)
    plt.legend()
    st.pyplot()

    # Create DataFrame to store results
    df_results = pd.DataFrame(ensemble_dict, index=decades)
    df_results.index.name = 'Decade'
    st.dataframe(df_results)
    

def plot_occupation_vs_model(job_data, occupation_data, model_types, data_sources):
    # Check if occupation_data is a DataFrame or a path to a CSV, and load if necessary
    if isinstance(occupation_data, str):
        occupation_data = pd.read_csv(occupation_data)

    # Collect unique jobs from occupation data
    unique_jobs = set(occupation_data['Occupation'])
    
    # Prepare data for plotting
    x_values = []  # Average ensemble values
    y_values = []  # Average occupation ratios
    job_labels = []  # Job names for labels
    
    for job in unique_jobs:
        ensemble_averages = []
        occupation_values = []

        # Calculate average ensemble values per decade
        for decade in range(1900, 2011, 10):
            decade_averages = []
            for source in data_sources:
                for model in model_types:
                    model_data = job_data.get((source, model))
                    if model_data is not None and job in model_data['job'].values:
                        data_field = 'P0_normalized_she'  # Adjust if using normalized_she
                        job_decade_data = model_data[(model_data['job'] == job) & (model_data['decade'] == decade)][data_field]
                        decade_averages.extend(job_decade_data.dropna().values)
            if decade_averages:
                average_decade_value = np.mean(decade_averages)
                ensemble_averages.append(average_decade_value)

        # Extract occupation data for the job
        job_occupation_data = occupation_data[occupation_data['Occupation'] == job]
        for decade in range(1900, 2011, 10):
            decade_occupation_data = job_occupation_data[job_occupation_data['Decade'] == decade]['Female']
            if not decade_occupation_data.empty:
                occupation_values.append(decade_occupation_data.mean())

        # Calculate the averages
        if ensemble_averages and occupation_values:
            avg_ensemble = np.nanmean(ensemble_averages)
            avg_occupation_ratio = np.nanmean(occupation_values)
            x_values.append(avg_occupation_ratio)
            y_values.append(avg_ensemble)
            job_labels.append(job)

    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x_values, y_values, alpha=0.7)
    plt.xlabel('Average Occupation Ratio for All Decades')
    plt.ylabel('Average Ensemble Results for All Decades')
    plt.title('Scatter Plot of Model Results vs Occupation Ratios')
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # Plot a diagonal line
    plt.plot([0, 1], [0, 1], 'r--')

    # Annotate points
    for i, txt in enumerate(job_labels):
        plt.annotate(txt, (x_values[i], y_values[i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.legend()

    
    st.pyplot()




def main():
    st.set_page_config(page_title="Data Visualizations", layout="wide")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title('Data Visualizations')

    data_sources = ['case_law', 'ny_times']  # List of all data sources
    model_types = ['bert-base-uncased', 'distilbert-base-uncased', 'albert-base-v2']
    graph_options = ["Normalized She Trend","Ensemble Comparison", "Jobs She Trend", "Jobs vs Occupation Relative Percentage","Cohens d", "Calculate Occupation Correlations", "Occupation Statistic Tests", "Scatter Plots", "P0_she Ratio Trend"]

    selected_data_sources = st.sidebar.multiselect("Select Data Sources", data_sources, default=data_sources)
    selected_model_types = st.sidebar.multiselect("Select Model Types", model_types, default=model_types)
    selected_graphs = st.sidebar.radio("Select Graphs to Display", graph_options)


    if "Normalized She Trend" == selected_graphs:
        normalized_data, P0_normalized_data, ensemble_data, P0_ensemble_data, base_results, base_P0_results, log_data, log_ensemble_data, base_log_results = load_normalized_data(selected_model_types, selected_data_sources)
        st.write("### Normalized She Trend")
        plot_normalized_she_trends(normalized_data, ensemble_data, base_results, selected_model_types, selected_data_sources)

        plot_P0_normalized_she_trends(P0_normalized_data, ensemble_data, base_P0_results, model_types, data_sources)
        st.write("### Normalized She Trend with Bars")
        plot_P0_base_model(base_results, model_types, data_sources)
        plot_P0_base_model(base_P0_results, model_types, data_sources)

    if "Ensemble Comparison" == selected_graphs:
        normalized_data, P0_normalized_data, ensemble_data, P0_ensemble_data, base_results, base_P0_results, log_data, log_ensemble_data, base_log_results = load_normalized_data(selected_model_types, selected_data_sources)
        st.write("### Ensemble Comparison")
        selected_data = st.sidebar.radio("Select Data to Use", ["normalized_she", "p0_she", "log_she"])
        if selected_data == 'normalized_she':
            selected_ensemble = ensemble_data
        elif selected_data == 'p0_she':
            selected_ensemble = P0_ensemble_data
        elif selected_data == 'log_she':
            selected_ensemble = log_ensemble_data

        plot_ensemble_comparison(selected_ensemble, selected_data_sources)
        compare_bias_scores(ensemble_data, log_ensemble_data, data_sources)

    if "Jobs She Trend" == selected_graphs:
        selected_data = st.sidebar.radio("Select Data to Use", ["normalized_she", "p0_she", "log_she"])
        if selected_data == 'normalized_she':
            data_field = 'normalized_she'
        elif selected_data == 'p0_she':
            data_field = 'P0_normalized_she'
        elif selected_data == 'log_she':
            data_field = 'log_prob_she'
        job_data = load_job_normalized_data(selected_data_sources, selected_model_types)
        visualize_job_normalized_data(job_data, selected_data_sources, selected_model_types, data_field)


    if "Jobs vs Occupation Relative Percentage" == selected_graphs:
        selected_data = st.sidebar.radio("Select Data to Use", ["normalized_she", "p0_she", "log_she"])
        if selected_data == 'normalized_she':
            data_field = 'normalized_she'
        elif selected_data == 'p0_she':
            data_field = 'P0_normalized_she'
        elif selected_data == 'log_she':
            data_field = 'log_prob_she'
        job_data = load_job_normalized_data(selected_data_sources, selected_model_types)
        occupation_data = pd.read_csv('data/occupation_decade_percentages_gender.csv')
        st.dataframe(occupation_data)
        visualize_job_normalized_data_with_occupation(job_data, selected_data_sources, selected_model_types, occupation_data, data_field)


    if "Calculate Occupation Correlations" == selected_graphs:
        selected_data = st.sidebar.radio("Select Data to Use", ["normalized_she", "p0_she", "log_she"])
        if selected_data == 'normalized_she':
            data_field = 'normalized_she'
        elif selected_data == 'p0_she':
            data_field = 'P0_normalized_she'
        elif selected_data == 'log_she':
            data_field = 'log_prob_she'
        job_data = load_job_normalized_data(selected_data_sources, selected_model_types)
        df_occupation_correlations = calculate_correlations_with_occupation(job_data, selected_data_sources, selected_model_types, data_field)
        st.write(df_occupation_correlations)
        df_ensemble_correlations = calculate_ensemble_correlations(job_data, selected_data_sources, selected_model_types, data_field)
        st.write(df_ensemble_correlations)

        df_occupation_correlations = df_occupation_correlations.dropna()
        merged_correlations = pd.merge(df_occupation_correlations, df_ensemble_correlations, left_on="Job", right_on="Job")
        st.dataframe(merged_correlations)

    if "Occupation Statistic Tests" == selected_graphs:
        selected_data = st.sidebar.radio("Select Data to Use", ["normalized_she", "p0_she", "log_she"])
        if selected_data == 'normalized_she':
            data_field = 'normalized_she'
        elif selected_data == 'p0_she':
            data_field = 'P0_normalized_she'
        elif selected_data == 'log_she':
            data_field = 'log_prob_she'
        job_data = load_job_normalized_data(selected_data_sources, selected_model_types)
        occupation_data = pd.read_csv('data/occupation_decade_percentages_gender.csv')
        anova_results = anova_job_data_with_occupation(job_data, data_sources, model_types, occupation_data, data_field)
        st.dataframe(anova_results)

    if "Scatter Plots" == selected_graphs:
        job_data = load_job_normalized_data(selected_data_sources, selected_model_types)
        occupation_data = pd.read_csv('data/occupation_decade_percentages_gender.csv')
        plot_occupation_vs_model(job_data, occupation_data, selected_model_types, selected_data_sources)
        

    if "P0_she Ratio Trend" == selected_graphs:
        calculate_log_probability_bias_score(selected_model_types, ['case_law'])
        calculate_log_probability_bias_score(selected_model_types, ['ny_times'])

    if "Cohens d" == selected_graphs:
        plot_cohens_d_trends(data_sources, model_types)

        


if __name__ == "__main__":
    main()

