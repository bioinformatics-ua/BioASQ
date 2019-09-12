import json
import yaml
from importlib import import_module
from logger import log
from collections import deque


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
