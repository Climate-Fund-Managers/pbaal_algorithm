import os
import time
from functools import wraps
from typing import Dict

import pandas as pd


def save_path(init):
    ts = time.time()
    @wraps(init)
    def wrapper(self,args: Dict):
        """
        Wrapper function to return the correct path name for saving models - to be used around constructor
        Arguments:
            args(Dict): Arguments dictionary as read from yaml file for all models
        """
        if args["run_active_learning"]:
            if not args['list_of_models']:
                unique_results_identifier = f"{args['model_name_or_path']}/non_active_one_model/{ts}"
            else:
                unique_results_identifier = f"{args['list_of_models'][0]}/non_active_majority/{ts}"
        else:
            if args["pool_based_learning"]:
                unique_results_identifier = f"{args['model_name_or_path']}/active_pool_based/{ts}"
            elif args["query_by_committee"]:
                unique_results_identifier = f"{args['list_of_models'][0]}/active_query_comittee/{ts}"
        
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


def create_save_path(init):
    @wraps(init)
    def wrapper(self,args:Dict):
        directory = f"{args['result_location']}/{args['unique_results_identifier']}/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        pd.DataFrame(args).to_csv(f"{directory}parameters.csv")
        init(self,args)
    return wrapper
