from models.model import ModelAPI
from utils import dynamicly_class_load
from os.path import exists, join
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from logger import log
from models.subnetworks.input_network import DetectionNetwork
from random import sample


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

    def is_training_data_in_cache(self, **kwargs):
        return exists(join(self.cache_folder, "{}_cache_traing_data.p".format(self.name)))

    def prepare_training_data(self, corpora, queries, retrieved, **kwargs):

        # positive document must be from previous module and training data
        positive_id = []
        for retrieved_train_data in retrieved["train"].values():
            positive_id.extend([x["id"] for x in retrieved_train_data["documents"] if x["id"] in queries.positive_ids])
        positive_id = set(positive_id)

        articles = {}
        # load corpus to memory
        for docs in corpora.read_documents_iterator():
            for doc in docs:
                articles[doc["id"]] = "{} {}".format(doc["title"], doc["abstract"])

        collection_ids = set(articles.keys())

        training_data = {}
        # select irrelevant and particly irrelevant articles
        for query_id, query_data in queries.train_data_dict.items():
            partially_positive_ids = set(map(lambda x: x["id"], retrieved["train"][query_id]["documents"]))
            retrieved_positive_ids = set(filter(lambda x: x in partially_positive_ids, query_data["documents"]))
            # irrelevant ids
            irrelevant_ids = (collection_ids-retrieved_positive_ids)-partially_positive_ids
            num_irrelevant_ids = 10*len(partially_positive_ids)
            num_irrelevant_ids = min(len(irrelevant_ids), num_irrelevant_ids)
            irrelevant_ids = sample(list(irrelevant_ids), num_irrelevant_ids)

            training_data[query_id] = {"positive_ids": retrieved_positive_ids,
                                       "partially_positive_ids": partially_positive_ids,
                                       "irrelevant_ids": irrelevant_ids}

        # total ids
        articles_ids = set()
        for data in training_data.values():
            for ids in data["positive_ids"]:
                articles_ids.add(ids)
            for ids in data["partially_positive_ids"]:
                articles_ids.add(ids)
            for ids in data["irrelevant_ids"]:
                articles_ids.add(ids)

        print("[DeepRank] Total ids selected for training {}".format(len(articles_ids)))
        log.info("[DeepRank] Total ids selected for training {}".format(len(articles_ids)))

        del articles

        exit(1)

        name = join(self.cache_folder, "{}_cache_traing_data.p".format(self.name))

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

    def build_train_arch(self, **kwargs):
        pass

    def train(self, simulation=False, **kwargs):
        steps = []
        model_output = None

        if "corpora" in kwargs:
            corpora = kwargs["corpora"]

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

        # prepare the training data
        if not self.is_training_data_in_cache(**kwargs):
            steps.append("[MISS] DeepRank training data in cache")
            if not simulation:
                self.prepare_training_data(**kwargs)
        else:
            steps.append("[READY] DeepRank training data in cache")

        # build the network
        if not self.is_trained():
            steps.append("[MISS] DeepRank build network")
            if not simulation:
                steps.append("[DEEPRANK] build network")
                self.build_network(**self.config)
                self.build_train_arch(**self.config)
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
