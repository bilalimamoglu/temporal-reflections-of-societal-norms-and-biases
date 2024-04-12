#!/opt/homebrew/bin/bash

# Get the path to the script's directory
SCRIPT_DIR=$(cd $(dirname "$0") && pwd)

# Setup logging
LOG_DIR="$SCRIPT_DIR/../data"
LOG_FILE="$LOG_DIR/download.log"
mkdir -p "$LOG_DIR"
echo "Starting download process..." > "$LOG_FILE"

# Directory for raw data
RAW_DATA_DIR="$SCRIPT_DIR/../data/raw"

# Ensure the raw data directories exist and create decade-specific directories
for year in 1900 1910 1920 1930 1940 1950 1960 1970 1980 1990 2000 2010; do
    mkdir -p "$RAW_DATA_DIR/case_law/$year"
    mkdir -p "$RAW_DATA_DIR/ny_times/$year"
done

# Convert Google Drive view links to gdown compatible links
convert_to_download_url() {
    echo "https://drive.google.com/uc?id=$1"
}

# Define URLs for the Google Drive files
declare -A CASE_LAW_URLS=(
    [1900]="$(convert_to_download_url '10kht4s9l1qV6D0JZrqT0Lw9muirYHkbN')"
    [1910]="$(convert_to_download_url '1FSPSSmSZpkmf-D62BpeOy85JfBOrNu4X')"
    [1920]="$(convert_to_download_url '1hS5TZFzWIIMD2lvZJHm7SqRDS2zuODHh')"
    [1930]="$(convert_to_download_url '1xrrBb6573rkbVhojqnt7QApLmsuj0fIs')"
    [1940]="$(convert_to_download_url '1WyuTXDFxv8tHCfTPizGJUR8FPz5jdSn4')"
    [1950]="$(convert_to_download_url '1586q5BnI9ELeJJtx-Zo9OuVKim2elpsV')"
    [1960]="$(convert_to_download_url '1gn9NajwZGwrhjmmWX9e72B88Sg9meSUi')"
    [1970]="$(convert_to_download_url '13DThhdim6vnrg7LmHsY0WJsXpG88r3yi')"
    [1980]="$(convert_to_download_url '1A4_TDs9GHxq-1dxU1lM0_o-IBFLy4h-N')"
    [1990]="$(convert_to_download_url '1ReJyK0yGYQCF_Nih4KelK_AFIQczdbYq')"
    [2000]="$(convert_to_download_url '1SV0AfSQ1EsTJVy_8d0VQQO_zv4t1Hlv9')"
    [2010]="$(convert_to_download_url '1YilB_i39jxBG7Nnrh_xJfrskhK0KWEGp')"
)

declare -A NY_TIMES_URLS=(
    [1900]="$(convert_to_download_url '10_Rd9vRFf4NIRHcdjqMZ4EsSadV4pyfC')"
    [1910]="$(convert_to_download_url '10YL1GEvlEeSsCuYcZK4XjFHXv_3kkxKM')"
    [1920]="$(convert_to_download_url '10b74Up9H2RXWm_56mEcJm3JfFjVwR5sk')"
    [1930]="$(convert_to_download_url '10atLgx_ThPvOSv9lpPSagem271ZGD-ft')"
    [1940]="$(convert_to_download_url '10aUFAR1aq4OsMAQJIUTCa6MsG1Y2-OeE')"
    [1950]="$(convert_to_download_url '10qDQZEZ9FhloxtZp6T6mAP5QfpELeGzD')"
    [1960]="$(convert_to_download_url '10loLsGea95wZic0OwXv7E3fpuHIQ5Gd9')"
    [1970]="$(convert_to_download_url '10cMwSEa2iiYWB4_XBR9JV5vUk3MTctWH')"
    [1980]="$(convert_to_download_url '10sDrBWh8ubpPX4UXLZMra1WxuTIX-7KK')"
    [1990]="$(convert_to_download_url '10rzGH6bw4yhNckiVEhvWcD0b5hPqxXJa')"
    [2000]="$(convert_to_download_url '10sF_IZQYFKI493xDg4EgS42VgcRXw6Gh')"
    [2010]="$(convert_to_download_url '10yYd1pXpuruVpX-0e-NVVopbDvMrkb6p')"
)

# Function to download datasets
download_data() {
    local data_type=$1
    local -n urls=$2
    local year file_url target_file

    for year in "${!urls[@]}"; do
        file_url="${urls[$year]}"
        target_file="$RAW_DATA_DIR/${data_type}/${year}/${data_type}_${year}.csv"
        echo "Downloading ${data_type} data for ${year}: $file_url" | tee -a "$LOG_FILE"
        gdown -O "$target_file" "$file_url" --fuzzy --quiet || echo "Failed to download $data_type data for $year" | tee -a "$LOG_FILE"
    done
}

# Execute downloads
#download_data "case_law" CASE_LAW_URLS
download_data "ny_times" NY_TIMES_URLS

echo "Data download complete." | tee -a "$LOG_FILE"