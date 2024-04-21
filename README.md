# temporal-reflections-of-societal-norms-and-biases


conda create -n my_project_env python=3.11
conda activate my_project_env



/opt/homebrew/bin/bash /Users/bilalimamoglu/repos/temporal-reflections-of-societal-norms-and-biases/scripts/download_data.sh


/opt/homebrew/bin/bash scripts/download_data.sh

#albert-base-v2
#bert-base-uncased
#distilbert-base-uncased

## Preprocessing
poetry run python scripts/preprocess_data.py --data_source ny_times --model_name albert-base-v2 --years_list 1900 1910 1920 1930 1940 1950 1960 1970 1980 1990 2000 2010 --reprocess
poetry run python scripts/preprocess_data.py --data_source ny_times --model_name bert-base-uncased --years_list 1900 1910 1920 1930 1940 1950 1960 1970 1980 1990 2000 2010 --reprocess
poetry run python scripts/preprocess_data.py --data_source ny_times --model_name distilbert-base-uncased --years_list 1900 1910 1920 1930 1940 1950 1960 1970 1980 1990 2000 2010 --reprocess


poetry run python scripts/preprocess_data.py --data_source case_law --model_name albert-base-v2 --years_list 1900 1910 1920 1930 1940 1950 1960 1970 1980 1990 2000 2010 --reprocess
poetry run python scripts/preprocess_data.py --data_source case_law --model_name bert-base-uncased --years_list 1900 1910 1920 1930 1940 1950 1960 1970 1980 1990 2000 2010 --reprocess
poetry run python scripts/preprocess_data.py --data_source case_law --model_name distilbert-base-uncased --years_list 1900 1910 1920 1930 1940 1950 1960 1970 1980 1990 2000 2010 --reprocess


## Training
poetry run python scripts/preprocess_data.py --data_source case_law --model_name bert-base-uncased --years_list 1900 1910 1920 1930 1940 1950 1960 1970 1980 1990 2000 2010 --reprocess
poetry run python scripts/preprocess_data.py --data_source case_law --model_name distilbert-base-uncased --years_list 1900 1910 1920 1930 1940 1950 1960 1970 1980 1990 2000 2010 --reprocess


## Training with Visualization Bars

poetry run python scripts/train_models.py --data_source ny_times --model_name albert-base-v2 --years_list 1900 1910 1920 1930 1940 1950 1960 1970 1980 1990 2000 2010 --max_steps 5000 --batch_size 8

poetry run python scripts/train_models.py --data_source ny_times --model_name bert-base-uncased --years_list 1900 1910 1920 1930 1940 1950 1960 1970 1980 1990 2000 2010 --max_steps 5000 --batch_size 8

poetry run python scripts/train_models.py --data_source ny_times --model_name distilbert-base-uncased --years_list 1900 1910 1920 1930 1940 1950 1960 1970 1980 1990 2000 2010 --max_steps 5000 --batch_size 8



## Alternative Trainings

Start-Process -FilePath "powershell" -ArgumentList "-NoExit", "-Command", "& {cd <path-to-your-scripts>; poetry run python scripts/train_models.py --data_source ny_times --model_name albert-base-v2 --years_list 2000 2010 --max_steps 5000 --batch_size 8 > albert_output.txt}" -WindowStyle Hidden


Start-Process -FilePath "powershell" -ArgumentList "-NoExit", "-Command", "& {cd <path-to-your-scripts>; poetry run python scripts/train_models.py --data_source ny_times --model_name bert-base-uncased --years_list 1900 1910 1920 1930 1940 1950 1960 1970 1980 1990 2000 2010 --max_steps 5000 --batch_size 8 > bert_output.txt}" -WindowStyle Hidden


Start-Process -FilePath "powershell" -ArgumentList "-NoExit", "-Command", "& {cd <path-to-your-scripts>; poetry run python scripts/train_models.py --data_source ny_times --model_name distilbert-base-uncased --years_list 1900 1910 1920 1930 1940 1950 1960 1970 1980 1990 2000 2010 --max_steps 5000 --batch_size 8 > distilbert_output.txt}" -WindowStyle Hidden




Start-Process -NoNewWindow -FilePath "powershell" -ArgumentList "cd '<path-to-your-scripts>'; poetry run python scripts/train_models.py --data_source ny_times --model_name albert-base-v2 --years_list 2000 2010 --max_steps 5000 --batch_size 8 > albert_output.txt"

Start-Process -NoNewWindow -FilePath "powershell" -ArgumentList "cd '<path-to-your-scripts>'; poetry run python scripts/train_models.py --data_source ny_times --model_name bert-base-uncased --years_list 1900 1910 1920 1930 1940 1950 1960 1970 1980 1990 2000 2010 --max_steps 5000 --batch_size 8 > bert_output.txt"

Start-Process -NoNewWindow -FilePath "powershell" -ArgumentList "cd '<path-to-your-scripts>'; poetry run python scripts/train_models.py --data_source ny_times --model_name distilbert-base-uncased --years_list 1900 1910 1920 1930 1940 1950 1960 1970 1980 1990 2000 2010 --max_steps 5000 --batch_size 8 > distilbert_output.txt"





## Streamlit Experiment Tracking

poetry run streamlit run scripts/visualize_pipeline.py


poetry run python scripts/check_cuda.py


poetry run python scripts/calculate_harness_results.py --data_source ny_times --years_list 1900 1910 1920 1930 1940 1950 1960 1970 1980 1990 --model_names albert-base-v2


poetry run python scripts/aggregate_harness_results.py --data_source ny_times --years_list 1900 --model_names albert-base-v2




## experiment


Start-Process -FilePath "powershell" -ArgumentList "-NoExit", "-Command", "& {cd <path-to-your-scripts>; poetry run python scripts/train_models.py --data_source ny_times --model_name bert-base-uncased --years_list 1900 --max_steps 5 --batch_size 8 > bert_output.txt}" -WindowStyle Hidden


start /b cmd /c "poetry run python scripts/train_models.py --data_source ny_times --model_name albert-base-v2 --years_list 1980 1990 2000 2010 --max_steps 5000 --batch_size 8 > albert_output.txt"

start /b cmd /c "poetry run python scripts/train_models.py --data_source ny_times --model_name distilbert-base-uncased --years_list 1910 1920 1930 1940 1950 1960 1970 1980 1990 2000 2010 --max_steps 5000 --batch_size 8 > distilbert_output.txt"

start /b cmd /c "poetry run python scripts/train_models.py --data_source ny_times --model_name bert-base-uncased --years_list 1900 1910 1920 1930 1940 1950 1960 1970 1980 1990 2000 2010 --max_steps 5000 --batch_size 8 > bert_output.txt"

poetry run python scripts/train_models.py --data_source ny_times --model_name bert-base-uncased --years_list 1900 --max_steps 25 --batch_size 8

poetry run python scripts/calculate_harness_results.py --data_source ny_times --years_list 1910 1920 1930 1940 1950 1960 1970 1980 --model_names albert-base-v2




start /b cmd /c "poetry run python scripts/train_models.py --data_source ny_times --model_name distilbert-base-uncased --years_list 1910 1920 1930 1940 1950 1960 1970 1980 1990 2000 2010 --max_steps 5000 --batch_size 8 > distilbert_output.txt"


poetry run python scripts/train_models.py --data_source ny_times --model_name distilbert-base-uncased --years_list 1900 1910 1920 1930 1940 1950 1960 1970 1980 1990 2000 2010 --max_steps 5000 --batch_size 8

poetry run python scripts/train_models.py --data_source ny_times --model_name albert-base-v2 --years_list 1980 1990 2000 2010 --max_steps 5000 --batch_size 8


poetry run python scripts/train_models.py --data_source ny_times --model_name bert-base-uncased --years_list 1990 2000 2010 --max_steps 5000 --batch_size 8

poetry run python scripts/calculate_harness_results.py --data_source ny_times --years_list 1990 2000 2010 --model_names bert-base-uncased

poetry run python scripts/train_models.py --data_source case_law --model_name albert-base-v2 --years_list 1900 1910 1920 1930 1940 1950 1960 1970 1980 1990 2000 2010 --max_steps 5000 --batch_size 8
