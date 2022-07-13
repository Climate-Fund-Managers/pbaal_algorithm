from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.

    """

    # Path to pretrained model or model identifier from huggingface.co/models
    model_name_or_path: str = field(default=None)  # microsoft/mpnet-base

    # Pretrained config name or path if not the same as model_name
    config_name: Optional[str] = field(default=None)

    # Pretrained tokenizer name or path if not the same as model_name
    tokenizer_name: Optional[str] = field(default=None)

    # Where do you want to store the pretrained models downloaded from huggingface.co
    cache_dir: str = field(default=None)

    # Whether to use one of the fast tokenizer (backed by the tokenizers' library) or not
    use_fast_tokenizer: bool = field(default=True)

    # Will use the token generated when running `transformers-cli login`
    # (necessary to use this script with private models)
    use_auth_token: bool = field(default=False)
