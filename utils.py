import json
import yaml
from importlib import import_module
from logger import log


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
