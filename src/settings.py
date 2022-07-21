
"""
Defintion of seeds that are set for training the models
"""
RANDOM_SEED = 12
SEED = 1


"""
Whether to use tensorboard for result reporting
"""
USE_TENSORBOARD = True


"""
Whether to run both training and evaluation steps
"""
DO_EVAL = True
DO_TRAIN = True


"""
When to log and evaluate models
"""
EVALUATION_STRATEGY = "steps"
EVAL_STEPS = 100

LOGGING_STRATEGY =  "steps"
LOGGING_STEPS =  100


"""
The name of the task
"""
TASK_NAME = "mnli"


"""
Format for output 
"""
TYPE_FILE = "csv"


