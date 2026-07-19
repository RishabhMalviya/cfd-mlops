.PHONY: help dataset

# Overridable from the command line, e.g.
#   make dataset CACHE_DIR=data/drivaer_data/cache_full DECIMATE=0
DATA_DIR         ?= ./data/drivaer_data
CACHE_DIR        ?= ./data/drivaer_data/cache_decimated_99
DECIMATE         ?= 1
TARGET_REDUCTION ?= 0.99
MAX_RUNS         ?= 50

ifeq ($(DECIMATE),1)
DECIMATE_FLAG := --decimate
else
DECIMATE_FLAG := --no-decimate
endif

help:
	@echo "make dataset [DATA_DIR=...] [CACHE_DIR=...] [DECIMATE=0|1] [TARGET_REDUCTION=0.99] [MAX_RUNS=50]"
	@echo "    Preprocess & cache the DrivAerML runs, then run the dataset smoke tests."

dataset:
	uv run python src/drivaer_dataset.py \
		--data-dir $(DATA_DIR) \
		--cache-dir $(CACHE_DIR) \
		$(DECIMATE_FLAG) \
		--target-reduction $(TARGET_REDUCTION) \
		--max-runs $(MAX_RUNS)
