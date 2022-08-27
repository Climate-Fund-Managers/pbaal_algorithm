import os
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
        if not args["training_method"]["run_active_learning"]:
            if not args['model']['list_of_models']:
                unique_results_identifier = f"{args['model']['model_name_or_path']}/non_active_one_model/{ts}"
            else:
                unique_results_identifier = f"{args['model']['list_of_models'][0]}/non_active_majority/{ts}"
        else:
            adapter_type = args["training_method"]["adapters"]["adapter_name"]
            if args["training_method"]["type"]:
                unique_results_identifier = f"{args['model']['model_name_or_path']}/active_pool_based/{adapter_type}/{ts}"
            else: 
                unique_results_identifier = f"{args['model']['list_of_models'][0]}/active_query_comittee/{adapter_type}/{ts}"
        
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
        list_of_models = args['model']["list_of_models"]
        if args['model']["multi_model"]:
            args['model']['model_name_or_path'] = list_of_models[0]
        init(self,args)
    return wrapper


def create_save_path(init):
    pd = SingletonBase().pandas
    @wraps(init)
    def wrapper(self,args:Dict):
        directory = f"{args['output']['result_location']}/{args['unique_results_identifier']}/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        pd.DataFrame(args).to_csv(f"{directory}parameters.csv")
        init(self,args)
    return wrapper


class SingletonBase:
    def __init__(self):
        self._modules = {}

    def __getattr__(self, module_name):
        if module_name not in self._modules:
            self._modules[module_name] = __import__(module_name)
        return self._modules[module_name]
