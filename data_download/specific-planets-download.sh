#!/bin/zsh

DATA_DIR="data/raw"
SPLIT="train"
planet_ids=("1010375142" "1024292144" "1029552010" "1031303815" "1042982756" "1047977648" "1048114509" "104891231" "1049092982")

suffixes=("AIRS-CH0_signal_0.parquet" "AIRS-CH0_signal_1.parquet" "FGS1_signal_0.parquet" "FGS1_signal_1.parquet")
calibration_folders=("AIRS-CH0_calibration_0" "AIRS-CH0_calibration_1" "FGS1_calibration_0" "FGS1_calibration_1")
calibration_files=("dark.parquet" "dead.parquet" "flat.parquet" "linear_corr.parquet" "read.parquet")

for planet_id in $planet_ids; do
  for suffix in $suffixes; do
    FILENAME="${SPLIT}/${planet_id}/${suffix}"
    DEST_PATH="${DATA_DIR}/${SPLIT}/${planet_id}"

    if [ -f "${DEST_PATH}/${suffix}" ]; then
      echo "File ${DEST_PATH}/${suffix} already exists. Skipping download."
      continue
    fi

    mkdir -p "$DEST_PATH"
    kaggle competitions download -f "$FILENAME" -p "$DEST_PATH" -c ariel-data-challenge-2025 || true
  done
  for folder in $calibration_folders; do
    for file in $calibration_files; do
      FILENAME="${SPLIT}/${planet_id}/${folder}/${file}"
      DEST_PATH="${DATA_DIR}/${SPLIT}/${planet_id}/${folder}"

      if [ -f "${DEST_PATH}/${file}" ]; then
        echo "File ${DEST_PATH}/${file} already exists. Skipping download."
        continue
      fi
      
      mkdir -p "$DEST_PATH"
      kaggle competitions download -f "$FILENAME" -p "$DEST_PATH" -c ariel-data-challenge-2025 || true
    done
  done
done