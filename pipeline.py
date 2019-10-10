from utils import yaml_loader, json_loader, dynamicly_class_load, reset_graph
from dataset.dataset import Corpora, Queries, TestQueries
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
        reset_graph()
        # first module input is always the corpora plus training queires
        next_module_input = {"corpora": self.corpora, "queries": self.queries, "steps": []}
        for module in self.modules:
            next_module_input = module.train(simulation=simulation, **next_module_input)

        return next_module_input

    def inference(self, simulation=False, query=None, queries_file=None):
        if query is not None:
            query = [{"query_id": "manual_submited", "query": query}]
            next_module_input = {"data_to_infer": query, "steps": []}
        if queries_file is not None:
            next_module_input = {"data_to_infer": TestQueries(queries_file), "steps": []}
        else:
            next_module_input = {"data_to_infer": self.queries, "steps": []}

        for module in self.modules:
            next_module_input = module.inference(simulation=simulation, train=False, **next_module_input)
            # small transformation TODO this can be avoid with a refactor
            next_module_input["data_to_infer"] = next_module_input["retrieved"]

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
