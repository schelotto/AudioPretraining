from datasets import load_dataset, concatenate_datasets, DatasetDict
from typing import List
from transformers import AutoProcessor, EncodecFeatureExtractor

import datasets


def combine_datasets(
        dataset_paths: List[str],
        split: str = "train",
        sample_field: str = "audio",
        validation_split_percentage: int = 10,
        seed: int = 42
) -> DatasetDict:
    """
    Combine multiple datasets from given paths, and split them into training and validation sets.

    Args:
        dataset_paths (List[str]): List of paths to the datasets.
        split (str): The dataset split to load (e.g., "train", "validation").
        sample_field (str): The field name to use for retrieving samples (e.g., "audio").
        validation_split_percentage (int): Percentage of data to use for validation split.
        seed (int): Seed for shuffling the dataset.

    Returns:
        DatasetDict: A dictionary containing the combined training and validation datasets.
    """
    datasets_splits = []

    for dataset_path in dataset_paths:
        dataset_split = load_dataset(dataset_path, split=split)
        datasets_splits.append(dataset_split)

    # Concatenate the datasets into a single dataset
    if len(datasets_splits) > 1:
        combined_dataset = concatenate_datasets(datasets_splits).shuffle(seed=seed)
    else:
        combined_dataset = datasets_splits[0]

    # Create training and validation splits
    num_validation_samples = len(combined_dataset) * validation_split_percentage // 100
    train_dataset = combined_dataset.select(range(num_validation_samples, len(combined_dataset)))
    validation_dataset = combined_dataset.select(range(num_validation_samples))

    # Build a DatasetDict
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": validation_dataset
    })

    # Extract only the relevant sample field
    def extract_samples(batch):
        return {sample_field: batch[sample_field]}

    dataset_dict = dataset_dict.map(extract_samples, batched=True)
    return dataset_dict


def prepare_datasets(
        dataset_dict: DatasetDict,
        feature_extractor,
        audio_column_name: str = "audio",
        max_duration_in_seconds: float = 10.0,
        min_duration_in_seconds: float = 1.0,
        preprocessing_num_workers: int = 8
) -> DatasetDict:
    """
    Prepare the datasets for training by applying feature extraction and filtering based on duration.

    Args:
        dataset_dict (DatasetDict): Dictionary containing the training and validation datasets.
        feature_extractor: The feature extractor to use for audio processing.
        audio_column_name (str): The column name containing the audio samples.
        max_duration_in_seconds (float): Maximum duration of audio samples in seconds.
        min_duration_in_seconds (float): Minimum duration of audio samples in seconds.
        preprocessing_num_workers (int): Number of workers to use for preprocessing.

    Returns:
        DatasetDict: A dictionary containing the processed training and validation datasets.
    """
    sampling_rate = feature_extractor.sampling_rate
    max_length = int(max_duration_in_seconds * sampling_rate)
    min_length = int(min_duration_in_seconds * sampling_rate)

    dataset_dict = dataset_dict.cast_column(
        audio_column_name, datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
    )

    def prepare_dataset(batch):
        sample = batch[audio_column_name]
        inputs = feature_extractor(
            sample['array'], sampling_rate=sampling_rate, max_length=max_length, truncation=True
        )
        batch["input_length"] = len(inputs.input_values)
        batch["input_values"] = inputs.input_values
        batch["padding_mask"] = inputs.padding_mask
        return batch

    vectorized_datasets = dataset_dict.map(
        prepare_dataset,
        num_proc=preprocessing_num_workers,
        remove_columns=dataset_dict["train"].column_names,
    )

    vectorized_datasets = vectorized_datasets.remove_columns("input_length")
    return vectorized_datasets


class EncodecPretrainingDataCollator:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def __call__(self, features):
        batch = self.feature_extractor.pad(
            features,
            padding="longest",
            return_tensors="pt",
        )
        return batch

