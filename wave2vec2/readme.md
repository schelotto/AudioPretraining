# Wav2Vec2 Pretraining

<img src="logo/wav2vec2.png" alt="Project Logo" width="300"/>

---

## Overview

This project focuses on pretraining the Wav2Vec2 model using unlabeled audio data. Wav2Vec2 is a powerful model architecture designed for speech representation learning, enabling downstream tasks like automatic speech recognition (ASR) with minimal supervision. This repository provides a modular and well-structured implementation for pretraining Wav2Vec2, leveraging the HuggingFace Transformers library.

### Description of Files

- **config/config.json**: Contains configurations for model training, including paths to the dataset, model parameters, and training hyperparameters.
- **src/arguments.py**: Defines `ModelArguments`, `DataArguments`, and `TrainingArguments` using dataclasses, making argument handling more structured and organized.
- **src/data_collator.py**: Implements a custom data collator that dynamically pads inputs and applies masking for self-supervised pretraining.
- **src/training.py**: Handles the core logic for loading data, initializing the model, and running the training loop using the `Trainer` class.
- **main.py**: The main script to start the training process, parsing configurations from command-line arguments or a JSON file.

## Requirements

Ensure you have the following dependencies installed:

- Python 3.7+
- PyTorch 1.7+
- Hugging Face Transformers 4.0+
- Datasets library 1.6+
- Other dependencies specified in `requirements.txt` (if applicable)

You can install the necessary Python packages using pip:

```bash
pip install torch transformers datasets

