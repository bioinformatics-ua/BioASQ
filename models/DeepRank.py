from models.model import ModelAPI
from utils import dynamicly_class_load
from os.path import exists, join
from logger import log
from models.subnetworks.input_network import InputNetwork


class DeepRank(ModelAPI):
    def __init__(self,
                 prefix_name,
                 cache_folder,
                 top_k,
                 tokenizer,
                 embedding,
                 input_network,
                 measure_network,
                 aggregation_network,
                 **kwargs):

        self.top_k = top_k
        self.cache_folder = cache_folder
        self.prefix_name = prefix_name

        # dynamicly load tokenizer object
        name, attributes = list(tokenizer.items())[0]
        _class = dynamicly_class_load("tokenizers."+name, name)
        self.tokenizer = _class.maybe_load(cache_folder=cache_folder, prefix_name=self.prefix_name, **attributes)

        # dynamicly load embedding Object
        name, attributes = list(embedding.items())[0]
        _class = dynamicly_class_load("embeddings."+name, name)
        self.embedding = _class.maybe_load(cache_folder=cache_folder, prefix_name=self.prefix_name, tokenizer=self.tokenizer, **attributes)

        # input network is always the same
        self.input_network = InputNetwork(embedding=self.embedding, **input_network)

        # measure network

        # name
        self.name = "DeepRank_{}_{}_{}_{}".format(self.input_network.Q,
                                                  self.input_network.P,
                                                  self.input_network.S,
                                                  self.embedding.name)

    def is_trained(self):
        return exists(join(self.cache_folder, self.name))

    def build_network(self):
        # Build neural net in a lazzy way (only if its needed)
        pass

    def train(self, simulation=False, **kwargs):
        steps = []
        model_output = None

        if "corpora" in kwargs:
            corpora = kwargs.pop("corpora")

        if not self.tokenizer.is_trained():
            steps.append("[MISS] Tokenizer for the DeepRank")
            if not simulation:
                # code to create tokenizer
                print("[START] Create tokenizer for DeepRank")
                self.tokenizer.fit_tokenizer_multiprocess(corpora.read_documents_iterator(mapping=lambda x: x["title"]+" "+x["abstract"]))
                print("[FINISHED] Tokenizer for DeepRank with", len(self.tokenizer.word_counts), "terms")
        else:
            steps.append("[READY] Tokenizer for the DeepRank")

        if not self.embedding.has_matrix():
            steps.append("[MISS] Embedding matrix for the tokenizer")
            if not simulation:
                self.embedding.build_matrix()
        else:
            steps.append("[READY] Embedding matrix for the tokenizer")

        if not self.is_trained():
            steps.append("[MISS] DeepRank build network and train")
            if not simulation:
                # code to create index
                raise NotImplementedError()
        else:
            steps.append("[READY] DeepRank trained network")

        if simulation:
            return steps
        else:
            return model_output

    def inference(self, simulation=False, **kwargs):
        return []
