from utils import yaml_loader, json_loader, dynamicly_class_load
from dataset.dataset import Corpora, Queries
from os.path import exists
from logger import log
from models.model import ModelAPI


class Pipeline(ModelAPI):
    def __init__(self, config_file, mode):
        # attributes
        self.mode = mode
        self.modules = []
        self.steps = []  # string list with the pipeline execution routine
        self.corpora = None
        self.config = self.load_config(config_file)

    def build(self):
        # validate mode
        if self.mode is not None and self.config["mode"] != self.mode:
            print("[CONFIG FILE] Error: mode in config is", self.config["mode"], "but the runtime option is", self.mode)
            log.error("[CONFIG FILE] Error: mode in config is {} but the runtime option is {}".format(self.config["mode"], self.mode))

        # check important parameters in the config (TODO validation and error checking)
        print("[CONFIG FILE] cache_folder path:", "OK" if exists(self.config["cache_folder"]) else "FAIL")
        print("[CONFIG FILE] doc_repository folder path:", "OK" if exists(self.config["corpora"]["folder"]) else "FAIL")
        print("[CONFIG FILE] queires train path file:", "OK" if exists(self.config["queries"]["train_file"]) else "FAIL")
        print("[CONFIG FILE] queires validation path file:", "OK" if exists(self.config["queries"]["validation_file"]) else "FAIL")

        # setup documents repository
        self.corpora = Corpora(**self.config["corpora"])

        if self.mode == "train":
            # setup Queries
            self.queries = Queries(self.mode, **self.config["queries"])

        # setup modules
        for module in self.config["pipeline"]:
            name, attributes = list(module.items())[0]
            _class = dynamicly_class_load("models."+name, name)
            m_instance = _class(cache_folder=self.config["cache_folder"], prefix_name=self.corpora.name, **attributes)
            m_instance.build()
            self.modules.append(m_instance)

    def train(self, simulation=False):
        # first module input is always the corpora plus training queires
        next_module_input = {"corpora": self.corpora, "queries": self.queries}
        steps = []
        for module in self.modules:
            if simulation:
                steps.extend(module.train(simulation=True, **next_module_input))
            else:
                next_module_input = module.train(**next_module_input)

        if simulation:
            return steps
        else:
            return next_module_input

    def load_config(self, config_file):
        # load config file
        loaders = [yaml_loader, json_loader]  # will try this loaders in a sequential order
        for loader in loaders:
            config = loader(config_file)
            if config is not None:
                print("[CONFIG FILE]", loader.__name__, "was used to load the configuration file")
                log.info("[CONFIG FILE] {} was used to load the configuration file".format(loader.__name__))
                break

        if config is None:
            print("[CONFIG FILE] Error: was not possible to read the configuration file")
            log.error("[CONFIG FILE] Error: was not possible to read the configuration file")
            exit(1)

        return config
