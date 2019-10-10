import json
import yaml
import h5py
from importlib import import_module
from logger import log
from collections import deque
from tensorflow import reset_default_graph, set_random_seed
from tensorflow.keras import backend as K
from metrics.evaluators import f_map, f_recall
import numpy as np
import random
import time


def yaml_loader(file_name):
    try:
        with open(file_name) as f:
            return yaml.safe_load(f)
    except Exception as e:
        log.warning(e)
    return None


def json_loader(file_name):
    try:
        with open(file_name) as f:
            return json.load(f)
    except Exception as e:
        log.warning(e)
    return None


def dynamicly_class_load(module_path, class_name):
    # reflex to load the class in runtime by name
    return getattr(import_module(module_path), class_name)


def config_to_string(pars):
    str = ""
    if isinstance(pars, dict):
        for k, v in pars.items():
            str += config_to_string(v)
    elif isinstance(pars, list):
        for e in pars:
            str += config_to_string(e)
    else:
        str += "{}_".format(pars)

    return str


class LimitedDict(dict):
    def __init__(self, limit, *args, **kw):
        super().__init__(*args, **kw)
        self.max_entry_limit = limit
        self.current_elments = 0
        # list where head is least freq and tail is the most frequent
        self.frequency_list = deque(maxlen=self.max_entry_limit)

    def __update_frequency(self, key):
        # move update
        self.frequency_list.remove(key)
        self.frequency_list.append(key)

    def __setitem__(self, key, value):
        if super().__contains__(key):
            self.__update_frequency(key)
            return

        if self.current_elments < self.max_entry_limit:
            super().__setitem__(key, value)
            self.frequency_list.append(key)  # add right (tail)
            self.current_elments += 1
        else:
            # free least used
            remove_key = self.frequency_list.popleft()  # pop left (head)
            super().__delitem__(remove_key)

            # add
            super().__setitem__(key, value)
            self.frequency_list.append(key)  # add right

    def __getitem__(self, key):
        self.__update_frequency(key)
        return super().__getitem__(key)


def save_model_weights(file_name, model):
    with h5py.File(file_name, 'w') as f:
        weight = model.get_weights()
        for i in range(len(weight)):
            f.create_dataset('weight'+str(i), data=weight[i])


def load_model_weights(file_name, model):
    with h5py.File(file_name, 'r') as f:
        weight = []
        for i in range(len(f.keys())):
            weight.append(f['weight'+str(i)][:])
        model.set_weights(weight)


def reset_graph(seed=42):
    K.clear_session()
    reset_default_graph()
    set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def evaluation(dict_results, gold_standard, top_k):

    start_eval_time = time.time()
    predictions = []
    expectations = []

    for _id in gold_standard.keys():
        expectations.append(gold_standard[_id])
        predictions.append(list(map(lambda x: x["id"], dict_results[_id]["documents"])))

    bioasq_map = f_map(predictions, expectations, bioASQ=True)
    str_bioasq_map = "[DEEPRANK] BioASQ MAP@10: {}".format(bioasq_map)
    print(str_bioasq_map)
    log.info(str_bioasq_map)
    str_map = "[DEEPRANK] Normal MAP@10: {}".format(f_map(predictions, expectations))
    print(str_map)
    log.info(str_map)
    str_recall = "[DEEPRANK] Normal RECALL@{}: {}".format(top_k, f_recall(predictions, expectations, at=top_k))
    print(str_recall)
    log.info(str_recall)
    log.info("Evaluation time {}".format(time.time()-start_eval_time))

    return bioasq_map
