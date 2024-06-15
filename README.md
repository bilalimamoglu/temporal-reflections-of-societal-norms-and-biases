# Temporal Reflections of Societal Norms and Biases

This repository contains scripts and resources for the thesis titled "Temporal Reflections of Societal Norms and Biases." The research investigates how transformer models process and replicate gender biases from historical texts, analyzing their temporal evolution and potential reflections on societal norms.

## Setup

To set up the environment, follow these steps:

1. Create and activate a Conda environment:
    ```bash
    conda create -n my_project_env python=3.11
    conda activate my_project_env
    ```

2. Install the necessary dependencies using the provided environment file:
    ```bash
    conda env create -f environment.yml
    ```

3. Download the necessary data:
    ```bash
    /opt/homebrew/bin/bash /Users/bilalimamoglu/repos/temporal-reflections-of-societal-norms-and-biases/scripts/download_data.sh
    ```

## Preprocessing

Preprocess the data for different models and datasets:

```bash
# NY Times Dataset
poetry run python scripts/preprocess_data.py --data_source ny_times --model_name albert-base-v2 --years_list 1900 1910 1920 1930 1940 1950 1960 1970 1980 1990 2000 2010 --reprocess
poetry run python scripts/preprocess_data.py --data_source ny_times --model_name bert-base-uncased --years_list 1900 1910 1920 1930 1940 1950 1960 1970 1980 1990 2000 2010 --reprocess
poetry run python scripts/preprocess_data.py --data_source ny_times --model_name distilbert-base-uncased --years_list 1900 1910 1920 1930 1940 1950 1960 1970 1980 1990 2000 2010 --reprocess

# Harvard Case Law Dataset
poetry run python scripts/preprocess_data.py --data_source case_law --model_name albert-base-v2 --years_list 1900 1910 1920 1930 1940 1950 1960 1970 1980 1990 2000 2010 --reprocess
poetry run python scripts/preprocess_data.py --data_source case_law --model_name bert-base-uncased --years_list 1900 1910 1920 1930 1940 1950 1960 1970 1980 1990 2000 2010 --reprocess
poetry run python scripts/preprocess_data.py --data_source case_law --model_name distilbert-base-uncased --years_list 1900 1910 1920 1930 1940 1950 1960 1970 1980 1990 2000 2010 --reprocess
```

## Training

Train the models on the preprocessed data:

```bash
# NY Times Dataset
poetry run python scripts/train_models.py --data_source ny_times --model_name albert-base-v2 --years_list 1900 1910 1920 1930 1940 1950 1960 1970 1980 1990 2000 2010 --max_steps 1250 --batch_size 32
poetry run python scripts/train_models.py --data_source ny_times --model_name bert-base-uncased --years_list 1900 1910 1920 1930 1940 1950 1960 1970 1980 1990 2000 2010 --max_steps 1250 --batch_size 32
poetry run python scripts/train_models.py --data_source ny_times --model_name distilbert-base-uncased --years_list 1900 1910 1920 1930 1940 1950 1960 1970 1980 1990 2000 2010 --max_steps 1250 --batch_size 32

# Harvard Case Law Dataset
poetry run python scripts/train_models.py --data_source case_law --model_name albert-base-v2 --years_list 1900 1910 1920 1930 1940 1950 1960 1970 1980 1990 2000 2010 --max_steps 1250 --batch_size 32
poetry run python scripts/train_models.py --data_source case_law --model_name bert-base-uncased --years_list 1900 1910 1920 1930 1940 1950 1960 1970 1980 1990 2000 2010 --max_steps 1250 --batch_size 32
poetry run python scripts/train_models.py --data_source case_law --model_name distilbert-base-uncased --years_list 1900 1910 1920 1930 1940 1950 1960 1970 1980 1990 2000 2010 --max_steps 1250 --batch_size 32
```


## Experiment Tracking

To track experiments and visualize results, use Streamlit:

```bash
poetry run streamlit run scripts/visualize_pipeline.py
```



## Additional Scripts

Various scripts are available for further analysis and experimentation:

```bash
# Check CUDA availability
poetry run python scripts/check_cuda.py

# Calculate harness results
poetry run python scripts/calculate_harness_results.py --data_source ny_times --years_list 1900 1910 1920 1930 1940 1950 1960 1970 1980 1990 --model_names albert-base-v2

# Aggregate harness results
poetry run python scripts/aggregate_harness_results.py --data_source ny_times --years_list 1900 --model_names albert-base-v2

# Train models with experimental settings
poetry run python scripts/train_models.py --data_source ny_times --model_name bert-base-uncased --years_list 1900 --max_steps 25 --batch_size 8
poetry run python scripts/calculate_harness_results.py --data_source ny_times --years_list 1910 1920 1930 1940 1950 1960 1970 1980 --model_names albert-base-v2

# Calculate unmasking probabilities
poetry run python scripts/calculate_unmasking_probabilities.py --data_source ny_times --years_list 1900 1910 1920 1930 1940 1950 1960 1970 1980 1990 2000 2010 --model_names albert-base-v2
poetry run python scripts/calculate_unmasking_probabilities.py --data_source ny_times --years_list 1900 1910 1920 1930 1940 1950 1960 1970 1980 1990 2000 2010 --model_names distilbert-base-uncased
poetry run python scripts/calculate_unmasking_probabilities.py --data_source ny_times --years_list 1900 1910 1920 1930 1940 1950 1960 1970 1980 1990 2000 2010 --model_names bert-base-uncased

poetry run python scripts/calculate_unmasking_probabilities.py --data_source case_law --years_list 1900 1910 1920 1930 1940 1950 1960 1970 1980 1990 2000 2010 --model_names albert-base-v2
poetry run python scripts/calculate_unmasking_probabilities.py --data_source case_law --years_list 1900 1910 1920 1930 1940 1950 1960 1970 1980 1990 2000 2010 --model_names distilbert-base-uncased
poetry run python scripts/calculate_unmasking_probabilities.py --data_source case_law --years_list 1900 1910 1920 1930 1940 1950 1960 1970 1980 1990 2000 2010 --model_names bert-base-uncased

# Aggregate unmasking results
poetry run python scripts/aggregate_unmasking_results.py --data_source case_law --model_name bert-base-uncased
poetry run python scripts/aggregate_unmasking_results.py --data_source case_law --model_name distilbert-base-uncased
poetry run python scripts/aggregate_unmasking_results.py --data_source case_law --model_name albert-base-v2
poetry run python scripts/aggregate_unmasking_results.py --data_source ny_times --model_name bert-base-uncased
poetry run python scripts/aggregate_unmasking_results.py --data_source ny_times --model_name distilbert-base-uncased
poetry run python scripts/aggregate_unmasking_results.py --data_source ny_times --model_name albert-base-v2
```


## Conclusion

This repository provides a comprehensive framework for exploring how transformer models reflect and perpetuate societal norms and biases, particularly gender biases. By following the steps and using the provided scripts, researchers can replicate and extend this study to further understand and mitigate biases in AI systems.

For more details on the methodology and findings, please refer to the thesis document.

## Future Work

There are several avenues for future research and enhancements based on the findings of this thesis:

1. **Extended Pronoun Set:** Incorporating a broader set of gender pronouns beyond 'he' and 'she' to capture a more comprehensive range of gender representations.
2. **Non-Binary Gender Inclusion:** Expanding the analysis to include non-binary and third-gender representations to understand AI's handling of diverse gender identities.
3. **Complex Linguistic Analysis:** Analyzing more complex linguistic elements such as adjectives and context-specific terms to uncover subtler forms of bias.
4. **Sophisticated Evaluation Metrics:** Utilizing advanced metrics like conference resolution for a more detailed analysis of biases.
5. **Large Language Models (LLMs):** Including state-of-the-art LLMs to enhance the understanding of biases and their mitigation.
6. **Broader Societal Biases:** Examining other societal biases such as race and age to provide a comprehensive view of the ethical challenges in AI.
7. **Technical Architecture Analysis:** Investigating how different architectural settings influence gender bias perception in transformer models.
8. **Bias Mitigation Strategies:** Developing and implementing effective bias mitigation techniques to ensure AI systems are fair and equitable.

## Acknowledgments

This project was made possible by the contributions of numerous individuals and organizations. Special thanks to the data providers and the open-source community for their tools and resources.

## License

This repository is licensed under the MIT License. See the LICENSE file for more information.

## Contact

For questions or further information, please contact Bilal Imamoglu at [imamogluubilal@gmail.com].

