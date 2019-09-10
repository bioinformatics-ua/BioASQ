from models.model import ModelAPI
from utils import dynamicly_class_load
from os.path import exists, join
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from logger import log
from models.subnetworks.input_network import DetectionNetwork
from random import sample, choice
import numpy as np
import pickle
import gc


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
        return exists(self.__training_data_file_name(**kwargs))

    def __training_data_file_name(self, origin, **kwargs):
        return join(self.cache_folder, "O{}_T{}_M{}_cache_traing_data.p".format(origin, self.tokenizer.name, self.name))

    def prepare_training_data(self, corpora, queries, retrieved, **kwargs):

        articles = {}
        # load corpus to memory
        for docs in corpora.read_documents_iterator():
            for doc in docs:
                articles[doc["id"]] = "{} {}".format(doc["title"], doc["abstract"])

        collection_ids = set(articles.keys())

        training_data = {}
        # select irrelevant and particly irrelevant articles
        for query_id, query_data in queries.train_data_dict.items():
            log.info("[DeepRank] Prepare query {}".format(query_id))
            partially_positive_ids = set(map(lambda x: x["id"], retrieved["train"][query_id]["documents"]))
            retrieved_positive_ids = set(filter(lambda x: x in partially_positive_ids, query_data["documents"]))

            if len(retrieved_positive_ids) == 0:
                log.warning("[DeepRank] Query {} does not have any positive articles, action=skip".format(query_id))
                # skip
                continue
            # irrelevant ids
            irrelevant_ids = (collection_ids-retrieved_positive_ids)-partially_positive_ids
            num_irrelevant_ids = 10*len(partially_positive_ids)
            num_irrelevant_ids = min(len(irrelevant_ids), num_irrelevant_ids)
            irrelevant_ids = sample(list(irrelevant_ids), num_irrelevant_ids)

            training_data[query_id] = {"positive_ids": retrieved_positive_ids,
                                       "partially_positive_ids": partially_positive_ids,
                                       "irrelevant_ids": irrelevant_ids,
                                       "query": self.tokenizer.tokenize_query(query_data["query"])}

        # total ids
        used_articles_ids = set()
        for data in training_data.values():
            for ids in data["positive_ids"]:
                used_articles_ids.add(ids)
            for ids in data["partially_positive_ids"]:
                used_articles_ids.add(ids)
            for ids in data["irrelevant_ids"]:
                used_articles_ids.add(ids)

        print("[DeepRank] Total ids selected for training {}".format(len(used_articles_ids)))
        log.info("[DeepRank] Total ids selected for training {}".format(len(used_articles_ids)))
        tokenized_articles = {id: self.tokenizer.tokenize_article(articles[id]) for id in used_articles_ids}

        del articles
        log.info("[DeepRank] Call garbage collector {}".format(gc.collect()))

        data = {"train": training_data, "articles": tokenized_articles}

        # save
        with open(self.__training_data_file_name(**kwargs), "wb") as f:
            pickle.dump(data, f, protocol=4)

        return data

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

    def build_train_arch(self, input_network, **kwargs):
        Q = input_network["Q"]
        P = input_network["P"]
        S = input_network["S"]

        query_token_input = Input(shape=(Q,), name="dr_query_tokens")
        positive_snippet_input = Input(shape=(Q, P, S), name="positive_snippet_tokens")
        positive_snippet_position_input = Input(shape=(Q, P), name="positive_snippet_position_tokens")
        negative_snippet_input = Input(shape=(Q, P, S), name="negative_snippet_tokens")
        negative_snippet_position_input = Input(shape=(Q, P), name="negative_snippet_position_tokens")

        positive_documents_score = self.deeprank_model([query_token_input, positive_snippet_input, positive_snippet_position_input])
        negative_documents_score = self.deeprank_model([query_token_input, negative_snippet_input, negative_snippet_position_input])

        inputs = [query_token_input, positive_snippet_input, positive_snippet_position_input, negative_snippet_input, negative_snippet_position_input]

        self.trainable_deep_rank = Model(inputs=inputs, outputs=[positive_documents_score, negative_documents_score], name="deep_rank_trainable_arch")

        # tensor loss
        p_loss = K.mean(K.maximum(0.0, 1.0 - positive_documents_score + negative_documents_score))
        self.trainable_deep_rank.add_loss(p_loss)
        self.trainable_deep_rank.summary(print_fn=log.info)

    def train(self, simulation=False, **kwargs):
        steps = kwargs["steps"]
        corpora = kwargs["corpora"]
        queries = kwargs["queries"]

        model_output = {"origin": self.name,
                        "queries": queries,
                        "steps": steps}

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
                train_data = self.prepare_training_data(**kwargs)
        else:
            steps.append("[READY] DeepRank training data in cache")
            if not simulation:
                # Load
                print("[DeepRank] Loading preprocessed training data")
                with open(self.__training_data_file_name(**kwargs), "rb") as f:
                    train_data = pickle.load(f)

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

        return model_output

    def inference(self, simulation=False, **kwargs):
        return []

    # DATA GENERATOR FOR THIS MODEL
    def training_generator(self, training_data, hyperparameters, **kwargs):

        # initializer
        batch_size = hyperparameters["batch_size"]
        query = []
        query_positive_doc = []
        query_positive_doc_position = []
        query_negative_doc = []
        query_negative_doc_position = []

        # generator loop
        while True:
            # stop condition
            if len(query) >= batch_size:
                query = np.array(query)
                p = np.random.permutation(query.shape[0])
                query = query[p]
                query_positive_doc = np.array(query_positive_doc)[p]
                query_positive_doc_position = np.array(query_positive_doc_position)[p]
                query_negative_doc = np.array(query_negative_doc)[p]
                query_negative_doc_position = np.array(query_negative_doc_position)[p]

                X = [query, query_positive_doc, query_positive_doc_position, query_negative_doc, query_negative_doc_position]
                yield X
                # reset
                query = []
                query_positive_doc = []
                query_positive_doc_position = []
                query_negative_doc = []
                query_negative_doc_position = []
            else:
                # select a random
                query_id, query_data = choice(training_data["train"].items())


class TrainDataGenerator(object):
    def __init__(self, hyperparameters, **kwargs):
        self.hyperparameters = hyperparameters
