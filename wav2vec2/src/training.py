import logging
import datasets

from datasets import DatasetDict, concatenate_datasets, load_dataset
from transformers import (
    Wav2Vec2Config,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForPreTraining,
    Trainer,
    TrainingArguments as HFTrainingArguments,
    set_seed,
    AdamW,
)
from .arguments import ModelArguments, DataArguments, TrainingArguments
from .data_collator import DataCollatorForWav2Vec2Pretraining

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_and_prepare_datasets(data_args: DataArguments, feature_extractor: Wav2Vec2FeatureExtractor) -> DatasetDict:
    datasets_splits = []
    for dataset_config_name, train_split_name in zip(data_args.dataset_config_names, data_args.dataset_split_names):
        dataset_split = load_dataset(
            data_args.dataset_name,
            dataset_config_name,
            split=train_split_name,
            cache_dir=data_args.cache_dir,
        )
        datasets_splits.append(dataset_split)

    raw_datasets = DatasetDict()
    if len(datasets_splits) > 1:
        raw_datasets["train"] = concatenate_datasets(datasets_splits).shuffle(seed=data_args.seed)
    else:
        raw_datasets["train"] = datasets_splits[0]

    num_validation_samples = raw_datasets["train"].num_rows * data_args.validation_split_percentage // 100
    raw_datasets["validation"] = raw_datasets["train"].select(range(num_validation_samples))
    raw_datasets["train"] = raw_datasets["train"].select(range(num_validation_samples, raw_datasets["train"].num_rows))

    raw_datasets = raw_datasets.cast_column(
        data_args.audio_column_name, datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
    )

    max_length = int(data_args.max_duration_in_seconds * feature_extractor.sampling_rate)
    min_length = int(data_args.min_duration_in_seconds * feature_extractor.sampling_rate)

    def prepare_dataset(batch):
        sample = batch[data_args.audio_column_name]
        inputs = feature_extractor(
            sample["array"], sampling_rate=sample["sampling_rate"], max_length=max_length, truncation=True
        )
        batch["input_values"] = inputs.input_values[0]
        batch["input_length"] = len(inputs.input_values[0])
        return batch

    vectorized_datasets = raw_datasets.map(
        prepare_dataset,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=raw_datasets["train"].column_names,
    )

    if min_length > 0.0:
        vectorized_datasets = vectorized_datasets.filter(
            lambda x: x > min_length,
            num_proc=data_args.preprocessing_num_workers,
            input_columns=["input_length"],
        )

    vectorized_datasets = vectorized_datasets.remove_columns("input_length")
    return vectorized_datasets


def create_model(model_args: ModelArguments) -> Wav2Vec2ForPreTraining:
    config = Wav2Vec2Config.from_pretrained(model_args.model_name_or_path)
    model = Wav2Vec2ForPreTraining.from_pretrained(model_args.model_name_or_path, config=config)
    if model_args.freeze_feature_extractor:
        model.freeze_feature_encoder()
    return model


def train(model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments):
    set_seed(training_args.seed)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_args.model_name_or_path)
    vectorized_datasets = load_and_prepare_datasets(data_args, feature_extractor)
    model = create_model(model_args)

    data_collator = DataCollatorForWav2Vec2Pretraining(
        model=model,
        feature_extractor=feature_extractor,
        pad_to_multiple_of=8,
        mask_time_prob=0.05,
        mask_time_length=10,
    )

    hf_training_args = HFTrainingArguments(
        output_dir=training_args.output_dir,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        evaluation_strategy="steps",
        logging_steps=training_args.logging_steps,
        save_steps=training_args.save_steps,
        num_train_epochs=training_args.num_train_epochs,
        seed=training_args.seed,
    )

    trainer = Trainer(
        model=model,
        args=hf_training_args,
        train_dataset=vectorized_datasets["train"],
        eval_dataset=vectorized_datasets["validation"],
        data_collator=data_collator,
        optimizers=(AdamW(model.parameters(), lr=5e-5), None),
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
