#!/bin/bash

BASE_URL="https://huggingface.co/datasets/neashton/drivaerml/resolve/main"
LOCAL_DIR="./data/drivaer_data"

download_run() {
  i=$1
  RUN_DIR="$LOCAL_DIR/run_$i"
  mkdir -p "$RUN_DIR"

  BASE="$BASE_URL/run_$i"

  wget -q --show-progress "$BASE/boundary_$i.vtp"           -O "$RUN_DIR/boundary_$i.vtp"
  wget -q --show-progress "$BASE/drivaer_$i.stl"            -O "$RUN_DIR/drivaer_$i.stl"
  wget -q --show-progress "$BASE/force_mom_$i.csv"          -O "$RUN_DIR/force_mom_$i.csv"
  wget -q --show-progress "$BASE/force_mom_constref_$i.csv" -O "$RUN_DIR/force_mom_constref_$i.csv"
  wget -q --show-progress "$BASE/geo_ref_$i.csv"            -O "$RUN_DIR/geo_ref_$i.csv"
  wget -q --show-progress "$BASE/geo_parameters_$i.csv"     -O "$RUN_DIR/geo_parameters_$i.csv"

  echo "Done: run_$i"
}

export -f download_run
export BASE_URL
export LOCAL_DIR

mkdir -p "$LOCAL_DIR"
seq 1 23 | xargs -P 5 -I {} bash -c 'download_run "$@"' _ {}  # 24 is corrupted, so we skip it
seq 25 50 | xargs -P 5 -I {} bash -c 'download_run "$@"' _ {}
echo "All downloads complete"
