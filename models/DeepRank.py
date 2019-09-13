from models.model import ModelAPI
from utils import dynamicly_class_load, config_to_string
from os.path import exists, join
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from logger import log
from models.subnetworks.input_network import DetectionNetwork
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adadelta
from random import sample, choice
from utils import LimitedDict
import time
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
                 evaluation,
                 **kwargs):

        self.top_k = top_k
        self.cache_folder = cache_folder
        self.prefix_name = prefix_name
        self.evaluation = evaluation  # TODO is not been used
        self.config = kwargs
        self.deeprank_model = None

        # dynamicly load tokenizer object
        name, attributes = list(tokenizer.items())[0]
        _class = dynamicly_class_load("tokenizers."+name, name)
        self.tokenizer = _class.maybe_load(cache_folder=cache_folder, prefix_name=self.prefix_name, **attributes)

        # dynamicly load embedding Object
        name, attributes = list(embedding.items())[0]
        _class = dynamicly_class_load("embeddings."+name, name)
        self.embedding = _class.maybe_load(cache_folder=cache_folder, prefix_name=self.prefix_name, tokenizer=self.tokenizer, **attributes)

        self.SNIPPET_POSITION_PADDING_VALUE = -1

        # some caching mechanism for inference
        self.cached_articles_tokenized = LimitedDict(2000000)
        self.cached_queries_tokenized = LimitedDict(50000)

        # name
        self.name = "DeepRank_{}_{}_{}".format(self.tokenizer.name, self.embedding.name, config_to_string(self.config))

    def is_trained(self):
        return exists(join(self.cache_folder, self.name))

    def is_training_data_in_cache(self, **kwargs):
        return exists(self.__training_data_file_name(**kwargs))

    def __training_data_file_name(self, origin, **kwargs):
        return join(self.cache_folder, "O{}_T{}_cache_traing_data.p".format(origin, self.tokenizer.name))

    def prepare_training_data(self, corpora, queries, retrieved, **kwargs):

        articles = {}
        # load corpus to memory
        for docs in corpora.read_documents_iterator():
            for doc in docs:
                articles[doc["id"]] = "{} {}".format(doc["title"], doc["abstract"])

        collection_ids = set(articles.keys())

        training_data = {}
        print("[DeepRank] Prepare the training data")
        # select irrelevant and particly irrelevant articles
        DEBUG_JUMP = True
        if not DEBUG_JUMP:
            for i, items in enumerate(queries.train_data_dict.items()):

                query_id, query_data = items
                log.info("[DeepRank] Prepare query {} id:{}".format(i, query_id))
                partially_positive_ids = set(map(lambda x: x["id"], retrieved["train"][query_id]["documents"]))
                retrieved_positive_ids = set(filter(lambda x: x in partially_positive_ids, query_data["documents"]))

                if len(retrieved_positive_ids) == 0:
                    log.warning("[DeepRank] Query {} does not have any positive articles, action=skip".format(query_id))
                    # skip
                    continue
                # irrelevant ids
                irrelevant_ids = (collection_ids-retrieved_positive_ids)-partially_positive_ids
                num_irrelevant_ids = 10000  # 5*len(partially_positive_ids)
                num_irrelevant_ids = min(len(irrelevant_ids), num_irrelevant_ids)
                irrelevant_ids = sample(list(irrelevant_ids), num_irrelevant_ids)

                training_data[query_id] = {"positive_ids": list(retrieved_positive_ids),
                                           "partially_positive_ids": list(partially_positive_ids),
                                           "irrelevant_ids": list(irrelevant_ids),
                                           "query": self.tokenizer.tokenize_query(query_data["query"])}

        # manual load checkpoint
        # checkpoint
        print("LOAD")
        with open(join(self.cache_folder, "prepere_data_checkpoint.p"), "rb") as f:
            training_data = pickle.laod(f)

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

        # same index id, article
        articles_ids = list(used_articles_ids)
        articles_texts = [articles[id] for id in articles_ids]

        # clear memory
        del articles
        print("[GC]", gc.collect())

        log.debug("[DeepRank] before multiprocess tokenization {} == {}".format(len(articles_ids), len(articles_texts)))
        tokenized_articles = self.tokenizer.tokenizer_multiprocess(articles_texts, mode="articles")
        log.debug("[DeepRank] after multiprocess tokenization {} == {}".format(len(articles_ids), len(tokenized_articles)))
        tokenized_articles = dict(zip(articles_ids, tokenized_articles))
        log.debug("[DeepRank] {} == {}".format(len(articles_ids), len(tokenized_articles)))


        log.info("[DeepRank] Call garbage collector {}".format(gc.collect()))

        data = {"train": training_data, "articles": tokenized_articles}

        # save
        with open(self.__training_data_file_name(**kwargs), "wb") as f:
            pickle.dump(data, f, protocol=4)

        return data

    def build_network(self, input_network, measure_network, aggregation_network, hyperparameters, **kwargs):

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

    def build_train_arch(self, input_network, hyperparameters, **kwargs):
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

        optimizer = get_optimizer(**hyperparameters["optimizer"])
        self.trainable_deep_rank.compile(optimizer=optimizer)

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
            self.train_network(training_data=train_data, validation_data=kwargs["retrieved"]["validation"], queries=queries, **self.config)

            # save current weights of the model
            self.deeprank_model.save_weights(join(self.cache_folder, "last_weights_{}.h5".format(self.name)))

        return model_output

    def inference(self, data_to_infer, simulation=False, **kwargs):
        """
        data_to_infer: dict {query_id: {query:<str>, documents:<list with documents>}}
        """
        steps = kwargs["steps"]
        model_output = {"origin": self.name,
                        "steps": steps,
                        "retrieved": {}
                        }

        # lazzy build
        if self.deeprank_model is None:
            if not simulation:
                self.build_network(**self.config)  # assignment occurs here

            name = join(self.cache_folder, "last_weights_{}.h5".format(self.name))
            if exists(name):
                print("LOAD FROM CACHE DeepRank weights")
                self.deeprank_model.load_weights(name)
            else:
                steps.append("[MISS] DeepRank weights")
                log.warning("[DeepRank] Missing weights for deeprank, it will use the random initialized weights")

        if not simulation:
            inference_generator = self.inference_generator(inference_data=data_to_infer, input_network=self.config["input_network"], **kwargs)
            for i, gen_data in enumerate(inference_generator):
                X, docs_ids, query_id, query = gen_data
                log.info("[DeepRank] inference for  {}-{}".format(i, query_id))
                scores = self.deeprank_model.predict(X)
                scores = map(lambda x: x[0], scores.tolist())
                merge_scores_ids = list(zip(docs_ids, scores))
                merge_scores_ids.sort(key=lambda x: -x[1])
                merge_scores_ids = merge_scores_ids[:self.top_k]
                log.info(merge_scores_ids)
                model_output["retrieved"][query_id] = {"query": query,
                                                       "documents": list(map(lambda x: x[0], merge_scores_ids))}

        return model_output

    def train_network(self, training_data, validation_data, queries, hyperparameters, **kwargs):
        print("[DeepRank] Start training")
        epochs = hyperparameters["epoch"]
        batch_size = hyperparameters["batch_size"]
        steps = max(1, len(training_data["train"])//batch_size)

        training_generator = self.training_generator(training_data, hyperparameters, **kwargs)

        # sub sample the validation set because to speed up training
        sub_set_validation_size = int(len(validation_data)*0.15)
        sub_set_validation = dict(sample(validation_data.items(), sub_set_validation_size))
        # build gold_standard # queries.validation_data_dict
        sub_set_validation_gold_standard = {}
        for key in sub_set_validation.keys():
            sub_set_validation_gold_standard[key] = queries.validation_data_dict[key]["documents"]

        dict(map(lambda x: (x["query_id"], x["documents"]), queries.validation_data))

        for epoch in range(epochs):
            loss_per_epoch = []
            for step in range(steps):

                X = next(training_generator)

                start = time.time()
                loss_per_epoch.append(self.trainable_deep_rank.train_on_batch(X))
                _train_line_info = "Step: {} | loss: {} | current max loss: {} | current min loss: {} | time: {}".format(step,
                                                                                                                         loss_per_epoch[-1],
                                                                                                                         np.max(loss_per_epoch),
                                                                                                                         np.min(loss_per_epoch),
                                                                                                                         time.time()-start)
                print(_train_line_info, end="\r")
                log.info(_train_line_info)

            if epoch % 10 == 0:
                # compute validation score!
                sub_set_validation_scores = self.inference(data_to_infer=sub_set_validation, **kwargs)["retrieved"]
                self.show_evaluation(sub_set_validation_scores, sub_set_validation_gold_standard)

    def show_evaluation(self, dict_results, gold_standard):
        print(dict_results)
        raise NotImplementedError()

    # DATA GENERATOR FOR THIS MODEL
    def training_generator(self, training_data, hyperparameters, input_network, **kwargs):
        Q = input_network["Q"]
        P = input_network["P"]
        S = input_network["S"]
        # initializer
        batch_size = hyperparameters["batch_size"]
        num_partilly_positives_samples = hyperparameters["num_partially_positive_samples"]
        num_negatives_samples = hyperparameters["num_negative_samples"]

        query = []
        query_positive_doc = []
        query_positive_doc_position = []
        query_negative_doc = []
        query_negative_doc_position = []

        list_keys = list(training_data["train"].keys())

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
                query_id = choice(list_keys)
                query_data = training_data["train"][query_id]

                # padding the query
                tokenized_query = pad_sequences([query_data["query"]], maxlen=Q, padding="post")[0]

                for j in range(num_partilly_positives_samples+num_negatives_samples):
                    positive_article_id = choice(query_data["positive_ids"])
                    positive_tokenized_article = training_data["articles"][positive_article_id]

                    positive_snippets, positive_snippets_position = self.__snippet_interaction(tokenized_query, positive_tokenized_article, Q, P, S)

                    if j < num_partilly_positives_samples:
                        partially_positive_article_id = choice(query_data["partially_positive_ids"])
                        partially_positive_tokenized_article = training_data["articles"][partially_positive_article_id]

                        negative_snippets, negative_snippets_position = self.__snippet_interaction(tokenized_query, partially_positive_tokenized_article, Q, P, S)
                    else:
                        negative_article_id = choice(query_data["irrelevant_ids"])
                        negative_tokenized_article = training_data["articles"][negative_article_id]

                        negative_snippets, negative_snippets_position = self.__snippet_interaction(tokenized_query, negative_tokenized_article, Q, P, S)

                    # add
                    # not efficient
                    query.append(tokenized_query)

                    # positive doc
                    query_positive_doc.append(positive_snippets)
                    query_positive_doc_position.append(positive_snippets_position)

                    # negative doc
                    query_negative_doc.append(negative_snippets)
                    query_negative_doc_position.append(negative_snippets_position)

    def inference_generator(self, inference_data, input_network, **kwargs):
        """
        inference_data: [{query_id: <int>, query: <str>, documents: <list {}>}]
        """
        Q = input_network["Q"]
        P = input_network["P"]
        S = input_network["S"]

        for query_id, query_data in inference_data.items():

            # clear
            query = []
            query_doc = []
            query_doc_position = []

            if query_id in self.cached_queries_tokenized:
                tokenized_query = self.cached_queries_tokenized[query_id]
            else:
                tokenized_query = self.tokenizer.tokenize_query(query_data["query"])
                tokenized_query = pad_sequences([tokenized_query], maxlen=Q, padding="post")[0]
                self.cached_queries_tokenized[query_id] = tokenized_query

            for doc_data in query_data["documents"]:

                if doc_data["id"] in self.cached_articles_tokenized:
                    tokenized_doc = self.cached_articles_tokenized[doc_data["id"]]
                else:
                    tokenized_doc = self.tokenizer.tokenize_article(doc_data["original"])
                    self.cached_articles_tokenized[doc_data["id"]] = tokenized_doc

                doc_snippets, doc_snippets_position = self.__snippet_interaction(tokenized_query, tokenized_doc, Q, P, S)

                query.append(tokenized_query)
                query_doc.append(doc_snippets)
                query_doc_position.append(doc_snippets_position)

            X = [np.array(query), np.array(query_doc), np.array(query_doc_position)]
            yield X, map(lambda x: {"id": x["id"], "original": x["original"]}, query_data["documents"]), query_id, query_data["query"]

    def __snippet_interaction(self, tokenized_query, tokenized_article, Q, P, S):

        snippets = []
        snippets_position = []

        half_size = S//2

        # O(n^2) complexity, probably can do better with better data struct TODO see if is worthit
        for query_token in tokenized_query:
            snippets_per_token = []
            snippets_per_token_position = []
            if query_token != 0:  # jump padded token
                for i, article_token in enumerate(tokenized_article):
                    if article_token == query_token:

                        lower_index = i-half_size
                        lower_index = max(0, lower_index)

                        higher_index = i+half_size
                        higher_index = min(len(tokenized_article), higher_index)

                        snippets_per_token.append(tokenized_article[lower_index:higher_index])
                        snippets_per_token_position.append(i)

            if len(snippets_per_token) == 0:  # zero pad
                snippets.append(np.zeros((P, S), dtype=np.int32))
                snippets_position.append(np.zeros((P), dtype=np.int32) + self.SNIPPET_POSITION_PADDING_VALUE)
                continue

            max_snippets_len = min(P, len(snippets_per_token))

            # snippets in matrix format
            # pad
            snippets_per_token = pad_sequences(snippets_per_token, maxlen=S, padding="post")
            # fill the gaps
            _temp = np.zeros((P, S), dtype=np.int32)
            _temp[:max_snippets_len] = snippets_per_token[:max_snippets_len]
            snippets.append(_temp)

            # snippets_position in matrix format
            # pad
            snippets_per_token_position = pad_sequences([snippets_per_token_position], maxlen=P, padding="post", value=self.SNIPPET_POSITION_PADDING_VALUE)[0]
            snippets_position.append(snippets_per_token_position)

        return snippets, snippets_position


def get_optimizer(name, learning_rate):
    if name.lower() == "adadelta":
        return Adadelta(lr=learning_rate)
