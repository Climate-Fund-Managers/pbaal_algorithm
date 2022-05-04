from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataTrainingArguments:
    """
    Arguments relating to what data we are going to use in the model for
    training and eval.

    """
    # The name of the task to train on
    task_name: str = field(default='mnli')

    # The maximum total input sequence length after tokenization
    max_seq_length: int = field(default=512)  # this in line with mpnet

    # Whether to overwrite the cached preprocessed datasets
    overwrite_cache: bool = field(default=False)

    # Whether to pad all samples to max_seq_length
    pad_to_max_length: bool = field(default=True)

    # For debugging purposes or quicker training, truncate the number of
    # training examples to this value if set
    max_train_samples: Optional[int] = field(default=None)

    # For debugging purposes or quicker training, truncate the number of
    # evaluation examples to this value if set
    max_eval_samples: Optional[int] = field(default=None)

    # For debugging purposes or quicker training, truncate the number of
    # prediction examples to this value if set
    max_predict_samples: Optional[int] = field(default=None)

    # A csv or a json file containing the training data
    train_file: str = field(default=None)

    # A csv or a json file containing the validation data
    validation_file: str = field(default=None)

    # A csv or a json file containing the test data
    test_file: str = field(default=None)
