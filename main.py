import sys

import yaml

from src.transformer_with_adapters import TransformerWithAdapters

if __name__ == "__main__":
    file_location = sys.argv[1]

    with open(file_location) as f:
        arguments = yaml.safe_load(f)

    train_transformer = TransformerWithAdapters(arguments)
    if arguments['training_method']['run_active_learning']:
        train_transformer.run_active_learning()
    else:
        train_transformer.run_standard_learning()
