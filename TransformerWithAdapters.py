import argparse
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
import datasets
import numpy as np
import torch
import transformers
from datasets import concatenate_datasets, load_dataset, load_metric
from scipy.stats import entropy
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    AdapterTrainer,
    TrainerCallback
)

from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from transformers.adapters import CompacterConfig
import yaml


class TransformerWithAdapters(object):

    def __init__(self, arguments: object):

        self.task_to_keys = {"mnli": ("premise", "hypothesis")}
        self.adapter_info = {"adapter_name": "compacter", "adapter_config": CompacterConfig()}
        self.logger = logging.getLogger(__name__)

        random.seed(arguments['random_seed'])

        self.hf_args = {
            "model_name_or_path": arguments['model_name_or_path'],
            "task_name": arguments['task_name'],
            "do_train": arguments['do_train'],
            "do_eval": arguments['do_eval'],
            "max_seq_length": arguments['max_seq_length'],
            "per_device_train_batch_size": arguments['per_device_train_batch_size'],
            "per_device_eval_batch_size": arguments['per_device_eval_batch_size'],
            "learning_rate": arguments['learning_rate'],
            "overwrite_output_dir": arguments['overwrite_output_dir'],
            "output_dir": arguments['output_dir'] + arguments['task_name'] + "/",
            "logging_strategy": arguments['logging_strategy'],
            "logging_steps": arguments['logging_steps'],
            "evaluation_strategy": arguments['evaluation_strategy'],
            "eval_steps": arguments['eval_steps'],
            "seed": arguments['seed'],
            "max_steps": arguments['max_steps'],
            # The next line is important to ensure the dataset labels are properly passed to the model
            "remove_unused_columns": arguments['remove_unused_columns'],
            "num_train_epochs": arguments['num_train_epochs']

        }

        # 684 = (5476 (samples) / 32 (batch size)) *  4 epochs
        # Using max_steps instead of train_epoch since we want all experiment to train for the same
        # number of iterations.

        if arguments['use_tensorboard']:
            self.self.hf_args.update(
                {
                    "logging_dir": "/tmp/" + arguments['task_name'] + "/tensorboard",
                    "report_to": "tensorboard",
                }
            )

        self.raw_datasets = load_dataset(arguments['type_file'],
                                         data_files={'train': arguments['train_file'],
                                                     'validation_matched': arguments['validation_file'],
                                                     'test_matched': arguments['test_file']})

        if arguments['adapter_config'] == 'default':
            self.adapter_config = CompacterConfig()
        else:
            # TO BE ADJUSTED LATER
            pass

        self.use_adapters = arguments['use_adapters']
        self.adapter_name = arguments['adapter_name']
        self.adaptive_learning = arguments['adaptive_learning']
        self.target_score = arguments['target_score']
        self.initial_train_dataset_size = arguments['initial_train_dataset_size']
        self.query_samples_count = arguments['query_samples_count']

    def run_active_learning(self):

        original_train_dataset = self.raw_datasets["train"]

        train_dataset = original_train_dataset.select(
            random.sample(
                range(original_train_dataset.num_rows),
                int(original_train_dataset.num_rows * self.initial_train_dataset_size),
            )
        )

        # fake unlabeled dataset
        unlabeled_dataset = original_train_dataset.filter(
            lambda s: s["idx"] not in train_dataset["idx"]
        )

        self.raw_datasets["train"] = train_dataset
        self.raw_datasets["test"] = unlabeled_dataset

        self.hf_args["do_predict"] = True

        current_score = -1

        while unlabeled_dataset.num_rows > 0 and current_score < self.target_score:
            self.logger.info(f'Training using {self.raw_datasets["train"].num_rows}')

            evaluation_metrics, test_predictions = self.__train()
            current_score = evaluation_metrics["eval_combined_score"]

            samples_entropy = TransformerWithAdapters.__calculate_entropy(test_predictions)
            samples_entropy = torch.topk(samples_entropy, self.query_samples_count)

            new_train_samples = unlabeled_dataset.select(samples_entropy.indices.tolist())

            extended_train_dataset = concatenate_datasets(
                [self.raw_datasets["train"], new_train_samples],
                info=original_train_dataset.info,
            )

            unlabeled_dataset = original_train_dataset.filter(
                lambda s: s["idx"] not in extended_train_dataset["idx"]
            )

            self.raw_datasets["train"] = extended_train_dataset
            self.raw_datasets["test"] = unlabeled_dataset

    @staticmethod
    def __calculate_entropy(logit):
        probability = torch.nn.Softmax(dim=1)(torch.from_numpy(logit))
        samples_entropy = entropy(probability.transpose(0, 1).cpu())
        samples_entropy = torch.from_numpy(samples_entropy)
        return samples_entropy

    @staticmethod
    def __train():
        return [0, 0]


if __name__ == "__main__":
    file_location = sys.argv[0]

    with open(file_location) as f:
        arguments = yaml.safe_load(f)
        
    train_transformer = TransformerWithAdapters(arguments)
    train_transformer.run_active_learning()
