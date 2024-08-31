from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"}
    )
    freeze_feature_extractor: bool = field(
        default=True,
        metadata={"help": "Whether to freeze the feature extractor layers during training."}
    )


@dataclass
class DataArguments:
    dataset_name: str = field(
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_names: List[str] = field(
        metadata={"help": "The configuration names of the dataset to use (via the datasets library)."}
    )
    dataset_split_names: List[str] = field(
        metadata={"help": "The names of the training data set splits to use (via the datasets library)."}
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "Column in the dataset that contains speech file path. Defaults to 'audio'"}
    )
    max_duration_in_seconds: float = field(
        default=5.0,
        metadata={"help": "Filter out audio files that are longer than `max_duration_in_seconds` seconds"}
    )
    min_duration_in_seconds: float = field(
        default=3.0,
        metadata={"help": "Filter out audio files that are shorter than `min_duration_in_seconds` seconds"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."}
    )
    validation_split_percentage: int = field(
        default=1,
        metadata={"help": "Percentage of training data that should be used for validation if no validation is present in dataset."}
    )


@dataclass
class TrainingArguments:
    output_dir: str = field(
        default="./wav2vec2_pretrained",
        metadata={"help": "Where to store the final model."}
    )
    per_device_train_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per device for the training dataloader."}
    )
    per_device_eval_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per device for the evaluation dataloader."}
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "Total number of training epochs to perform."}
    )
    logging_steps: int = field(
        default=500,
        metadata={"help": "Number of steps between each logging."}
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "Number of steps between each save."}
    )
    seed: int = field(
        default=42,
        metadata={"help": "A seed for reproducible training."}
    )
