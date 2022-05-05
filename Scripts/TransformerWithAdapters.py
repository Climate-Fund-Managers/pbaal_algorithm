import logging
import os
import random
import sys
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
from transformers.adapters import CompacterConfig
import yaml
import DataTrainingArguments
import ModelArguments


class AdapterDropTrainerCallback(TrainerCallback):

    def __init__(self, adapter_name):
        self.adapter_name = adapter_name

    def on_step_begin(self, args, state, control, **kwargs):
        skip_layers = list(range(np.random.randint(0, 11)))  # TO CHANGE
        kwargs['model'].set_active_adapters(self.adapter_name, skip_layers=skip_layers)

    def on_evaluate(self, args, state, control, **kwargs):
        # Deactivate skipping layers during evaluation (otherwise it would use the
        # previous randomly chosen skip_layers and thus yield results not comparable
        # across different epochs)
        kwargs['model'].set_active_adapters(self.adapter_name, skip_layers=None)


class TransformerWithAdapters:
    # To dynamically drop adapter layers during training, we make use of HuggingFace's `TrainerCallback'.

    def __init__(self, args):

        self.task_to_keys = {"mnli": ("premise", "hypothesis")}
        self.logger = logging.getLogger(__name__)

        random.seed(args['random_seed'])

        self.hf_args = {
            "model_name_or_path": args['model_name_or_path'],
            "task_name": args['task_name'],
            "do_train": args['do_train'],
            "do_eval": args['do_eval'],
            "max_seq_length": args['max_seq_length'],
            "per_device_train_batch_size": args['per_device_train_batch_size'],
            "per_device_eval_batch_size": args['per_device_eval_batch_size'],
            "learning_rate": args['learning_rate'],
            "overwrite_output_dir": args['overwrite_output_dir'],
            "output_dir": args['output_dir'] + args['task_name'] + "/",
            "logging_strategy": args['logging_strategy'],
            "logging_steps": args['logging_steps'],
            "evaluation_strategy": args['evaluation_strategy'],
            "eval_steps": args['eval_steps'],
            "seed": args['seed'],
            "max_steps": args['max_steps'],
            # The next line is important to ensure the dataset labels are properly passed to the model
            "remove_unused_columns": args['remove_unused_columns'],
            "num_train_epochs": args['num_train_epochs']

        }

        # 684 = (5476 (samples) / 32 (batch size)) *  4 epochs
        # Using max_steps instead of train_epoch since we want all experiment to train for the same
        # number of iterations.

        if args['use_tensorboard']:
            self.hf_args.update(
                {
                    "logging_dir": "/tmp/" + args['task_name'] + "/tensorboard",
                    "report_to": "tensorboard",
                }
            )

        self.raw_datasets = load_dataset(args['type_file'],
                                         data_files={'train': args['train_file'],
                                                     'validation_matched': args['validation_file'],
                                                     'test_matched': args['test_file']})

        if args['adapter_config'] == 'default':
            self.adapter_config = CompacterConfig()
        else:
            # TO BE ADJUSTED LATER
            pass

        self.use_adapters = args['use_adapters']
        self.adapter_name = args['adapter_name']
        self.adaptive_learning = args['adaptive_learning']
        self.target_score = args['target_score']
        self.initial_train_dataset_size = args['initial_train_dataset_size']
        self.query_samples_count = args['query_samples_count']

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

    def __train(self):
        global train_dataset
        parser = HfArgumentParser(
            (ModelArguments, DataTrainingArguments, TrainingArguments))

        # parse command-line args into instances of the specified dataclass types
        if self.hf_args is not None:
            model_args, data_args, training_args = parser.parse_dict(self.hf_args)
        else:
            model_args, data_args, training_args = parser.parse_args_into_dataclasses()

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        log_level = training_args.get_process_log_level()
        self.logger.setLevel(log_level)
        datasets.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

        # Log on each process the small summary:
        self.logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
            + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )
        self.logger.info(f"Training/evaluation parameters {training_args}")

        # determine whether to start training from a checkpoint or newly
        last_checkpoint = None
        if (
                os.path.isdir(training_args.output_dir)
                and training_args.do_train
                and not training_args.overwrite_output_dir
        ):
            last_checkpoint = get_last_checkpoint(training_args.output_dir)  # returns a checkpoint with max number
            if last_checkpoint is None and len(os.listdir(
                    training_args.output_dir)) > 0:  # there is a non-empty output directory, we need to overwrite it
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif (
                    last_checkpoint is not None and training_args.resume_from_checkpoint is None
                    # checkpoint exists but we do not define ourselves which one to use
            ):
                self.logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )

        # set an initial seed to training
        set_seed(training_args.seed)

        # obtain the labels
        label_list = self.raw_datasets["train"].features["label"].names
        num_labels = len(label_list)

        # loading pre-trained model & tokenizer & config
        config = AutoConfig.from_pretrained(
            model_args.config_name
            if model_args.config_name
            else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name
            if model_args.tokenizer_name
            else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,  # microsoft/mpnet-base
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        if self.use_adapters:
            """CUSTOM CHANGES USING ADAPTERS"""
            # Add a new adapter
            model.add_adapter(self.adapter_name, config=self.adapter_config)
            # model.train_adapter(adapter_name) # we freeze all transformer parameters except for the parameters of
            # adapter model.add_classification_head(adapter_name, num_labels=num_labels) # add new classification
            # head on top since transformer layers are frozen
            model.set_active_adapters(self.adapter_name)  # registers the adapter as a default for training

        # set defaults for key names
        sentence1_key, sentence2_key = self.task_to_keys[data_args.task_name]

        # Padding strategy
        if data_args.pad_to_max_length:
            padding = "max_length"
        else:
            # We will pad later, dynamically at batch creation, to the max sequence length in each batch
            padding = False

        # Some models have set the order of the labels to use, so let's make sure we do use it.
        label_to_id = None
        if (
                model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
                and data_args.task_name is not None
        ):
            # Some have all caps in their config, some don't.
            label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
            if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
                label_to_id = {
                    i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)
                    # making sure the order is aligned
                }
            else:
                self.logger.warning(
                    "Your model seems to have been trained with labels, but they don't match the dataset: ",
                    f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                    "\nIgnoring the model labels as a result.",
                )
        elif data_args.task_name is None:
            label_to_id = {v: i for i, v in enumerate(label_list)}

        # now create also mapping from ids to labels
        if label_to_id is not None:
            model.config.label2id = label_to_id
            model.config.id2label = {id: label for label, id in config.label2id.items()}
        elif data_args.task_name is not None:
            model.config.label2id = {l: i for i, l in enumerate(label_list)}
            model.config.id2label = {id: label for label, id in config.label2id.items()}

        # define the max length of the sequence as min of model_max_length and max_seq_length
        if data_args.max_seq_length > tokenizer.model_max_length:
            self.logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

        def preprocess_function(examples):
            # Tokenize the texts
            args = ((examples[sentence1_key], examples[sentence2_key]))
            result = tokenizer(
                *args, padding=padding, max_length=max_seq_length, truncation=True)

            return result

        # preprocess/ tokenize dataset
        with training_args.main_process_first(desc="dataset map pre-processing"):
            raw_datasets = self.raw_datasets.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                # if overwrite True, do not load previously cached file
            )

        # set training dataset
        if training_args.do_train:
            if "train" not in raw_datasets:
                raise ValueError("--do_train requires a train dataset")

            train_dataset = raw_datasets["train"]
            # if we set limit on data to be used for testing, pick some data at random to use
            if data_args.max_train_samples is not None:
                train_dataset = train_dataset.select(range(data_args.max_train_samples))

        # set evaluation dataset
        if training_args.do_eval:
            if "validation_matched" not in raw_datasets:
                raise ValueError("--do_eval requires a validation dataset")
            eval_dataset = raw_datasets["validation_matched"]
            # if we set limit on data to be used for testing, pick some data at random to use
            if data_args.max_eval_samples is not None:
                eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

        # set test dataset
        if (
                training_args.do_predict
                or data_args.task_name is not None
                or data_args.test_file is not None
        ):
            if "test_matched" not in raw_datasets:
                raise ValueError("--do_predict requires a test dataset")

            predict_dataset = raw_datasets["test_matched"]
            # if we set limit on data to be used for testing, pick some data at random to use
            if data_args.max_predict_samples is not None:
                predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

        # Get the metric function
        metric = load_metric("glue", data_args.task_name)  # so for mnli glue metrics

        # Takes an `EvalPrediction` object (a namedtuple with a predictions and label_ids field)
        # and has to return a dictionary string to float.
        def compute_metrics(p: EvalPrediction):
            predicted = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            predicted = np.argmax(predicted, axis=1)

            result = metric.compute(predictions=predicted, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()

            return result

        # create objects that will form a batch by using a list of dataset elements as input
        if data_args.pad_to_max_length:
            data_collator = default_data_collator
        else:
            data_collator = None

        # Initialize our Trainer
        if self.use_adapters:
            trainer = AdapterTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset if training_args.do_train else None,
                eval_dataset=eval_dataset if training_args.do_eval else None,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
                data_collator=data_collator
            )

        else:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset if training_args.do_train else None,
                eval_dataset=eval_dataset if training_args.do_eval else None,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
                data_collator=data_collator
            )

        if self.adaptive_learning:
            trainer.add_callback(AdapterDropTrainerCallback(self.adapter_name))

        # define number of samples either as length of dataset or max_train_samples if performing tests
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )

        metrics_prefix = f"train_size_{min(max_train_samples, len(train_dataset))}_4e_all"

        # Training\
        # first check if previous chekpoint exists, otherwise start training from the scratch
        if training_args.do_train:
            checkpoint = None
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            elif last_checkpoint is not None:
                checkpoint = last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            metrics = train_result.metrics

            trainer.save_model()  # Saves the tokenizer too for easy upload

            trainer.log_metrics(metrics_prefix + "_train_metrics", metrics)
            trainer.save_metrics(metrics_prefix + "_train_metrics", metrics)
            trainer.save_state()

        # Evaluation
        evaluation_metrics = {}
        if training_args.do_eval:
            self.logger.info("*** Evaluate ***")

            # Loop to handle MNLI double evaluation (matched, mis-matched)
            tasks = [data_args.task_name]
            eval_datasets = [eval_dataset]
            tasks.append("mnli-mm")
            eval_datasets.append(raw_datasets["validation_mismatched"])

            for eval_dataset, task in zip(eval_datasets, tasks):
                metrics = trainer.evaluate(eval_dataset=eval_dataset)

                max_eval_samples = (
                    data_args.max_eval_samples  # in case we do testing with subsample
                    if data_args.max_eval_samples is not None
                    else len(eval_dataset)
                )
                metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

                trainer.log_metrics(metrics_prefix + "eval_metrics", metrics)
                trainer.save_metrics(metrics_prefix + "eval_metrics", metrics)

                evaluation_metrics = metrics

        test_predictions = None
        if training_args.do_predict:
            self.logger.info("*** Predict ***")

            # As we eval, loop to handle MNLI double evaluation (matched, mis-matched)
            tasks = [data_args.task_name]
            predict_datasets = [predict_dataset]
            tasks.append("mnli-mm")
            predict_datasets.append(raw_datasets["test_mismatched"])

            for predict_dataset, task in zip(predict_datasets, tasks):
                # Removing the `label` columns because it contains -1 and Trainer won't like that.
                predict_dataset = predict_dataset.remove_columns("label")
                test_predictions = trainer.predict(
                    predict_dataset, metric_key_prefix=metrics_prefix + "_predict_metrics"
                ).predictions

        return evaluation_metrics, test_predictions


if __name__ == "__main__":
    file_location = sys.argv[1]

    with open(file_location) as f:
        arguments = yaml.safe_load(f)

    train_transformer = TransformerWithAdapters(arguments)
    train_transformer.run_active_learning()
