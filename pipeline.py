from utils import yaml_loader, json_loader, dynamicly_class_load
from dataset.dataset import Corpora, Queries
from os.path import exists


class Pipeline:
    def __init__(self, config_file, mode, logging):
        # attributes
        self.logging = logging
        self.mode = mode
        self.modules = []
        self.steps = []  # string list with the pipeline execution routine
        self.corpora = None
        self.config = self.load_config(config_file)

    def build(self):
        # validate mode
        if self.mode is not None and self.config["mode"] != self.mode:
            print("[CONFIG FILE] Error: mode in config is", self.config["mode"], "but the runtime option is", self.mode)
            self.logging.error("[CONFIG FILE] Error: mode in config is", self.config["mode"], "but the runtime option is", self.mode)

        # check important parameters in the config (TODO validation and error checking)
        print("[CONFIG FILE] cache_folder path:", "OK" if exists(self.config["cache_folder"]) else "FAIL")
        print("[CONFIG FILE] doc_repository folder path:", "OK" if exists(self.config["corpora"]["folder"]) else "FAIL")
        print("[CONFIG FILE] queires train path file:", "OK" if exists(self.config["queries"]["folder"]+"_train.json") else "FAIL")
        print("[CONFIG FILE] queires validation path file:", "OK" if exists(self.config["queries"]["folder"]+"_validation.json") else "FAIL")

        # setup documents repository
        self.corpora = Corpora(**self.config["corpora"], logging=self.logging)

        if self.mode == "train":
            # setup Queries
            self.queries = Queries(mode=self.mode, **self.config["queries"])

        # setup modules
        for module in self.config["pipeline"]:
            name, attributes = list(module.items())[0]
            _class = dynamicly_class_load("models."+name, name)
            m_instance = _class(cache_folder=self.config["cache_folder"], prefix_name=self.corpora.name, **attributes)
            m_instance.build()
            self.modules.append(m_instance)

    def train(self):

        # first module input is always the corpora plus training queires
        next_module_input = {"corpora": self.corpora, "queries": self.queries}
        for module in self.modules:
            next_module_input = module.train(**next_module_input)

    def check_train_routine(self):
        steps = []
        for module in self.modules:
            steps.extend(module.train(simulation=True))
        self.__print_routine(steps)

    def check_inference_routine(self):
        steps = []
        for module in self.modules:
            steps.extend(module.inference(simulation=True))
        self.__print_routine(steps)

    def __print_routine(self, steps):
        print("\nRoutine")
        for step in steps:
            print(step)
        print("\n")

    def load_config(self, config_file):
        # load config file
        loaders = [yaml_loader, json_loader]  # will try this loaders in a sequential order
        for loader in loaders:
            config = loader(config_file, self.logging)
            if config is not None:
                print("[CONFIG FILE]", loader.__name__, "was used to load the configuration file")
                self.logging.info("[CONFIG FILE]", loader.__name__, "was used to load the configuration file")
                break

        if config is None:
            print("[CONFIG FILE] Error: was not possible to read the configuration file")
            self.logging.error("[CONFIG FILE] Error: was not possible to read the configuration file")
            exit(1)

        return config
