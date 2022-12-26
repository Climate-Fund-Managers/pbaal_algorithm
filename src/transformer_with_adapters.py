import json
import logging
import os
import random
import sys

import datasets
import numpy as np
import torch
import transformers
from datasets import (
    ClassLabel,
    Features,
    Value,
    concatenate_datasets,
    load_dataset,
    load_metric,
)
from scipy.stats import entropy
from transformers import (
    AdapterTrainer,
    AutoAdapterModel,
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
)
from transformers.adapters import (
    AdapterConfig,
    CompacterConfig,
    MAMConfig,
    PfeifferInvConfig,
    PrefixTuningConfig,
)
from transformers.trainer_utils import get_last_checkpoint

from src.model_args import ModelArguments
from src.settings import (
    DO_EVAL,
    DO_TRAIN,
    EVAL_STEPS,
    EVALUATION_STRATEGY,
    LOGGING_STEPS,
    LOGGING_STRATEGY,
    RANDOM_SEED,
    SEED,
    TASK_NAME,
    TYPE_FILE,
    USE_TENSORBOARD,
)
from src.trainer_callback import AdapterDropTrainerCallback
from src.training_args import DataTrainingArguments
from src.utils import SingletonBase, create_save_path, save_path, set_initial_model


class TransformerWithAdapters:
    # To dynamically drop adapter layers during training, we make use of HuggingFace's `TrainerCallback'.

    @save_path
    @set_initial_model
    @create_save_path
    def __init__(self, args):

        self.task_to_keys = {"mnli": ("premise", "hypothesis")}
        self.logger = logging.getLogger(__name__)
        self.pandas_importer = SingletonBase()

        random.seed(RANDOM_SEED)

        self.hf_args = {
            "model_name_or_path": args['model']['model_name_or_path'],
            "task_name": TASK_NAME,
            "do_train": DO_TRAIN,
            "do_eval": DO_EVAL,
            "max_seq_length": args['hyperparameters']['max_seq_length'],
            "per_device_train_batch_size": args['hyperparameters']['per_device_train_batch_size'],
            "per_device_eval_batch_size": args['hyperparameters']['per_device_eval_batch_size'],
            "learning_rate": args['hyperparameters']['learning_rate'],
            "overwrite_output_dir": args['output']['overwrite_output_dir'],
            "output_dir": f"{args['output']['output_dir']}/{args['unique_results_identifier']}/" ,
            "logging_strategy": LOGGING_STRATEGY,
            "logging_steps": LOGGING_STEPS,
            "evaluation_strategy": EVALUATION_STRATEGY,
            "eval_steps": EVAL_STEPS,
            "seed": SEED,
            # The next line is important to ensure the dataset labels are properly passed to the model
            "remove_unused_columns": args['training_method']['remove_unused_columns'],
            'num_train_epochs': args['hyperparameters']['num_train_epochs']

        }

        if USE_TENSORBOARD:
            self.hf_args.update(
                {
                    "logging_dir": "/tmp/" + TASK_NAME + "/tensorboard",
                    "report_to": "tensorboard",
                }
            )

        features = Features({'hypothesis': Value(dtype='string', id=None),
                             'idx': Value(dtype='int64', id=None),
                             'label': ClassLabel(num_classes=2, names=['entailment', 'neutral'],
                                                 id=None),
                             'premise': Value(dtype='string', id=None)})

        self.raw_datasets = load_dataset(TYPE_FILE,
                                         data_files={'train': args['data']['train_file'],
                                                     'validation_matched': args['data']['validation_file'],
                                                     'test_matched': args['data']['test_file']},
                                         features=features,
                                         encoding='cp1252')

        if not args['training_method']['run_active_learning']:
            self.raw_datasets["test"] = self.raw_datasets["test_matched"]

        config = {'compacter': CompacterConfig(),
                  'bottleneck_adapter':  AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu"),
                  'lang_adapter': PfeifferInvConfig(),
                  'prefix_tuning': PrefixTuningConfig(flat=False, prefix_length=30),
                  'mam_adapter': MAMConfig()
                  }

        if args["training_method"]["adapters"]['adapter_config'] == 'default':
            self.adapter_config = config[args["training_method"]["adapters"]['adapter_name']]

        else:
            # TO BE ADJUSTED LATER
            pass

        self.use_adapters = args["training_method"]["adapters"]['use_adapters']
        self.use_pretrained_adapters = args["training_method"]["adapters"]['use_pretrained_adapters']
        self.pretrained_adapter = args["training_method"]["adapters"]['pretrained_adapter']
        self.adapter_name = args["training_method"]["adapters"]['adapter_name']
        self.adaptive_learning = args["training_method"]["adapters"]['adaptive_learning']
        self.target_score = args["training_method"]['target_score']
        self.initial_train_dataset_size = args['hyperparameters']['initial_train_dataset_size']
        self.do_query = args["training_method"]['do_query']
        self.query_samples_count = args["training_method"]['query_samples_count']
        self.query_samples_ratio = args["training_method"]['query_samples_ratio']
        self.result_location = f"{args['output']['result_location']}/{args['unique_results_identifier']}/"
        self.type = args["training_method"]["type"]
        self.list_of_models = args["model"]['list_of_models']
        print(args["unique_results_identifier"])

    def run_majority_vote(self):
        pd = self.pandas_importer.pandas

        self.logger.info("MAJORITY VOTE LEARNING INITIATED")
        self.hf_args["do_predict"] = True
        self.logger.info(f'Training using full dataset')

        results= pd.DataFrame()
        for model in self.list_of_models:
            self.hf_args['model_name_or_path'] = model
            evaluation_metrics, test_predictions = self.__train()
            results[model] = test_predictions

        results['mean'] = results.mean(axis=1)
        results.round({'mean': 0})
        results['truth'] = self.raw_datasets['test_matched']['label']
        results['score'] = results['truth'] == results['mean']
        results.DataFrame(test_predictions).to_csv(self.result_location + 'ensemble_predictions.csv')

    def run_standard_learning(self):
        pd = self.pandas_importer.pandas

        self.logger.info("STANDARD LEARNING INITIATED")
        self.hf_args["do_predict"] = True
        self.logger.info(f'Training using full dataset')
        evaluation_metrics, test_metrics, test_predictions = self.__train()

        with open(self.result_location+'evaluation_metrics.json', 'w') as f:
            json.dump(evaluation_metrics, f)

        with open(self.result_location+'test_metrics.json', 'w') as f:
            json.dump(test_metrics, f)

        pd.DataFrame(test_predictions).to_csv(self.result_location+'predictions.csv')

    def run_active_learning(self):
        self.logger.info("ACTIVE LEARNING INITIATED")
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

        if self.type == "pool_based_learning":
            self.__pool_based_learning(original_train_dataset, unlabeled_dataset)
        elif self.type == "query_by_committee":
            self.__query_by_committee(original_train_dataset, unlabeled_dataset)

    def __query_by_committee(self, original_train_dataset, unlabeled_dataset):
        pd = self.pandas_importer.pandas

        current_score = -1
        all_scores = {"scores": [],
                      "# of records used": []}

        while unlabeled_dataset.num_rows > self.query_samples_count and current_score < self.target_score:

            self.logger.info(f'Query by committee training using {self.raw_datasets["train"].num_rows}')
            results = pd.DataFrame()

            for model in self.list_of_models:
                self.hf_args['model_name_or_path'] = model
                _, _, test_predictions = self.__train()
                results[model] = self.__get_predictions(test_predictions)

            results['variance'] = results.var(axis=1)

            if self.do_query:
                idxs = results['variance'].nlargest(self.query_samples_count).index.tolist()
            else:
                idxs = results['variance'].nlargest(int(len(results)*self.query_samples_ratio)).index.tolist()

            results['mean'] = results.mean(axis=1)
            results.round({'mean': 0})
            results['truth'] = unlabeled_dataset['label']
            results['score'] = results['truth'] == results['mean'] 
            current_score = results['score'].mean()
            all_scores['scores'].append(current_score)
            all_scores['# of records used'].append(self.raw_datasets["train"].num_rows)

            new_train_samples = unlabeled_dataset.select(idxs)

            extended_train_dataset = concatenate_datasets(
                [self.raw_datasets["train"], new_train_samples],
                info=original_train_dataset.info,
            )

            unlabeled_dataset = original_train_dataset.filter(
                lambda s: s["idx"] not in extended_train_dataset["idx"]
            )

            self.raw_datasets["train"] = extended_train_dataset
            self.raw_datasets["test"] = unlabeled_dataset

        # change, using wrong dataset
        pd.DataFrame(all_scores).to_csv(self.result_location + 'scores_per_run.csv')
        pd.DataFrame({'idx': unlabeled_dataset['idx'],
                      'prediction': results['mean']}).to_csv(self.result_location+'predictions.csv')


    def __pool_based_learning(self, original_train_dataset, unlabeled_dataset):
        pd = self.pandas_importer.pandas
        current_score_eval = -1
        all_scores = {"scores_eval": [],
                      "scores_test": [],
                      "# of records used": []}

        while unlabeled_dataset.num_rows > self.query_samples_count and current_score_eval < self.target_score:

            self.logger.info(f'Training using {self.raw_datasets["train"].num_rows}')

            evaluation_metrics, test_metrics, test_predictions = self.__train()
            current_score_eval = evaluation_metrics["eval_accuracy"]
            all_scores['scores_eval'].append(current_score_eval)

            current_score_test = test_metrics["eval_accuracy"]
            all_scores['scores_test'].append(current_score_test)

            all_scores['# of records used'].append(self.raw_datasets["train"].num_rows)

            samples_entropy_all = TransformerWithAdapters.__calculate_entropy(test_predictions)
            if self.do_query:
                samples_entropy = torch.topk(samples_entropy_all, self.query_samples_count)
            else:
                samples_entropy = torch.topk(samples_entropy_all, int(unlabeled_dataset.num_rows*self.query_samples_ratio))

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

        test_predictions = self.__get_predictions(test_predictions)

        pd.DataFrame(all_scores).to_csv(self.result_location+'scores_per_run.csv')
        

    def __get_predictions(self, test_predictions):
        """
        Function to get the prediction from an N*M dimensional array
        Arguments:
            test_predictions (torch.array): N*M dimensional array of predictions
        Returns:
            (torch.array): N dimensional array of predicted values in range of 0 to n_classes
        """
        return torch.argmax(torch.nn.Softmax(dim=1)(torch.from_numpy(test_predictions)),dim = 1)

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
            # revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name
            if model_args.tokenizer_name
            else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            # revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        """CUSTOM CHANGES USING ADAPTERS"""
        if self.use_adapters:
            model = AutoAdapterModel.from_pretrained(
                model_args.model_name_or_path,  # microsoft/mpnet-base
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                # revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )

            "WHETHER TO USE PRETRAINED ADAPTER OR NOT"
            if self.use_pretrained_adapters:
                pretrained_adapter_name = self.pretrained_adapter.split('/')[1].split("-")[0]
                if pretrained_adapter_name != self.model_name_or_path.split("-")[0]:
                    raise Exception("Pretrained adapter was trained on different transformer than the one in use")
                else:
                    adapter_name = model.load_adapter(self.pretrained_adapter, source="hf")
                    model.active_adapters = adapter_name

            else:
                model.add_adapter(self.adapter_name, config=self.adapter_config)
                model.add_classification_head(self.adapter_name, num_labels=num_labels)
                model.train_adapter(self.adapter_name)
                model.set_active_adapters(self.adapter_name)  # registers the adapter as a default for training
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                # revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )

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
                load_from_cache_file=False,
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

            predict_dataset = raw_datasets["test"]
            test_dataset = raw_datasets['test_matched']

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
        # first check if previous checkpoint exists, otherwise start training from the scratch
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
            test_datasets = [test_dataset]
            tasks.append("mnli-mm")

            for predict_dataset, task in zip(predict_datasets, tasks):
                # Removing the `label` columns because it contains -1 and Trainer won't like that.
                predict_dataset = predict_dataset.remove_columns("label")
                test_predictions = trainer.predict(
                    predict_dataset, metric_key_prefix=metrics_prefix + "_predict_metrics"
                ).predictions

            # Equally calculate metrics on test data
            for test_dataset, task in zip(test_datasets, tasks):
                metrics = trainer.evaluate(eval_dataset=test_dataset)

                max_eval_samples = len(test_dataset)

                metrics["eval_samples"] = min(max_eval_samples, len(test_dataset))

                trainer.log_metrics(metrics_prefix + "eval_metrics", metrics)
                trainer.save_metrics(metrics_prefix + "eval_metrics", metrics)

                test_metrics = metrics

        return evaluation_metrics, test_metrics, test_predictions

