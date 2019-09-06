from models.model import ModelAPI
from utils import dynamicly_class_load
from os.path import exists, join
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from logger import log
from models.subnetworks.input_network import DetectionNetwork


class DeepRank(ModelAPI):
    def __init__(self,
                 prefix_name,
                 cache_folder,
                 top_k,
                 tokenizer,
                 embedding,
                 **kwargs):

        self.top_k = top_k
        self.cache_folder = cache_folder
        self.prefix_name = prefix_name
        self.config = kwargs

        # dynamicly load tokenizer object
        name, attributes = list(tokenizer.items())[0]
        _class = dynamicly_class_load("tokenizers."+name, name)
        self.tokenizer = _class.maybe_load(cache_folder=cache_folder, prefix_name=self.prefix_name, **attributes)

        # dynamicly load embedding Object
        name, attributes = list(embedding.items())[0]
        _class = dynamicly_class_load("embeddings."+name, name)
        self.embedding = _class.maybe_load(cache_folder=cache_folder, prefix_name=self.prefix_name, tokenizer=self.tokenizer, **attributes)

        # name
        self.name = self.get_name(**kwargs)

    def get_name(self, input_network, **kwargs):
        return "DeepRank_{}_{}_{}_{}".format(input_network["Q"],
                                             input_network["P"],
                                             input_network["S"],
                                             self.embedding.name)

    def is_trained(self):
        return exists(join(self.cache_folder, self.name))

    def build_network(self, input_network, measure_network, aggregation_network, **kwargs):

        # build 3 sub models
        detection_network = DetectionNetwork(embedding=self.embedding, **input_network)

        # measure_network
        name, attributes = list(measure_network.items())[0]
        _class = dynamicly_class_load("models.subnetworks."+name, name)
        measure_network = _class(**input_network, **attributes)

        # aggregation_network
        name, attributes = list(aggregation_network.items())[0]
        _class = dynamicly_class_load("models.subnetworks."+name, name)
        aggregation_network = _class(embedding_layer=detection_network.embedding_layer, **attributes)


        # assemble the network
        query_input = Input(shape=(detection_network.Q,), name="query_input")
        snippets_input = Input(shape=(detection_network.Q, detection_network.P, detection_network.S), name="snippets_input")
        snippets_position_input = Input(shape=(detection_network.Q, detection_network.P), name="snippets_position_input")

        deeprank_detection = detection_network([query_input, snippets_input])
        deeprank_measure = measure_network([deeprank_detection, snippets_position_input])
        deeprank_score = aggregation_network([query_input, deeprank_measure])

        self.deeprank_model = Model(inputs=[query_input, snippets_input, snippets_position_input], outputs=[deeprank_score])
        detection_network.summary(print_fn=log.info)
        measure_network.summary(print_fn=log.info)
        aggregation_network.summary(print_fn=log.info)
        self.deeprank_model.summary(print_fn=log.info)

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
            steps.append("[MISS] DeepRank build network")
            if not simulation:
                steps.append("[DEEPRANK] build network")
                self.build_network(**self.config)
        else:
            steps.append("[READY] DeepRank trained network")

        if not simulation:
            # train the network
            raise NotImplementedError("train the network")

        if simulation:
            return steps
        else:
            return model_output

    def inference(self, simulation=False, **kwargs):
        return []
