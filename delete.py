# This is a file to test some python code, does not belong with the project
from utils import yaml_loader

config = yaml_loader("config_example.yaml")

dp_config = config["pipeline"][1]["DeepRank"]


def config_to_string(pars):
    str = ""
    for k, v in pars.items():
        if isinstance(v, dict):
            str += config_to_string(v)
        if isinstance(v, list):
            for e in v:
                print(e)
                str += config_to_string(e)
        else:
            str += "{}_".format(v)
    return str

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

print(config_to_string(dp_config["measure_network"]))
