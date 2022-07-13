import numpy as np
from transformers import TrainerCallback


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
