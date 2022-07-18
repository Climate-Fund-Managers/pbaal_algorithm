import time
from functools import wraps
from typing import Dict


def save_path(init):
    ts = time.time()
    @wraps(init)
    def wrapper(self,args: Dict):
        """
        Wrapper function to return the correct path name for saving models - to be used around constructor
        Arguments:
            args(Dict): Arguments dictionary as read from yaml file for all models
        """
        first_model = args['list_of_models'][0]
        if args["pool_based_learning"]:
            unique_results_identifier = f"{args['model_name_or_path']}/active_pool_based/{ts}"
        elif args["query_by_committee"]:
            unique_results_identifier = f"{first_model}/active_query_comittee/{ts}"
        else: 
            unique_results_identifier = f"{first_model}/non_active/{ts}"
        args["unique_results_identifier"] = unique_results_identifier
        init(self,args)
    return wrapper

def set_initial_model(init):
    @wraps(init)
    def wrapper(self,args:Dict):
        """
        Wrapper function to set the correct initial model
        Arguments:
            args(Dict): Arguments dictionary as read from yaml file for all models
        """
        list_of_models = args["list_of_models"]
        if list_of_models:
            args['model_name_or_path'] = list_of_models[0]
        init(self,args)
    return wrapper
