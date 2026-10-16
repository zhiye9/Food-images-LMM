# Food Image Nutrition Estimation with Open-Weight Large Multimodal Models
This repository contains code for the manuscript "Food Image Nutrition Estimation with Open-Weight Large Multimodal Models: Fine-Tuning, Multi-View Ensembles, and Retrieval Augmentation". This project explores the application of open-weight Large Multimodal Models (LMMs) for food image nutrition estimation, with three learning paradigms (zero-shot benchmarks, supervised fine-tuning (SFT), and retrieval-augmented generation (RAG)). Using Gemma3-4B/12B and Qwen3-VL-4B/8B on two datasets, this study evaluated multiple configurations with a hierarchical diagnostic of knowledge, identification, and portion estimation.

![](figures/pipeline.png)

## Data availability
This project is based on nutrition images from [Nutrition5k](https://huggingface.co/datasets/mmathys/food-nutrients) and [ACETADA](https://skynet.ecn.purdue.edu/~coburn6/ACETADA/).

## Model access
This project uses [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct), [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), [Gemma-3-4B-IT](https://huggingface.co/google/gemma-3-4b-it), and [Gemma-3-12B-IT](https://huggingface.co/google/gemma-3-12b-it) from Hugging Face Hub.

# Installation
To install the required dependencies, run the following command:

```bash
python -m pip install -r requirements.txt
```

# Data Preparation
Expected data structure:

```text
ACETADA-HF-dataset.csv
ACETADA/<participant_id>/<before image>

nutrition5k_dataset/
  metadata/dish_metadata_cafe1.csv
  dish_ids/splits/rgb_train_ids.txt
  dish_ids/splits/rgb_test_ids.txt
  frames_sampled60/camera_A_frame_001.jpeg ...

food-nutrients/metadata.jsonl

FNDDS/
  2021-2023 FNDDS At A Glance - Foods and Beverages.xlsx
  2021-2023 FNDDS At A Glance - FNDDS Nutrient Values.xlsx
  2021-2023 FNDDS At A Glance - Portions and Weights.xlsx
```

Running following command after preparing the data:

```bash
python src/prepare_data.py --config src/config.yaml
```
