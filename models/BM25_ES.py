from models.model import ModelAPI
from utils import dynamicly_class_load
from os.path import exists, join
from elasticsearch import Elasticsearch, helpers
from metrics.evaluators import f_map, f_recall
import json
from logger import log
import pickle


class BM25_ES(ModelAPI):
    def __init__(self, prefix_name, cache_folder, top_k, address, tokenizer, evaluation=False):
        self.top_k = top_k
        self.cache_folder = cache_folder
        self.prefix_name = prefix_name
        self.evaluation = evaluation
        self.es = Elasticsearch([address])

        name, attributes = list(tokenizer.items())[0]
        _class = dynamicly_class_load("tokenizers."+name, name)
        self.tokenizer = _class.maybe_load(cache_folder=cache_folder, prefix_name=self.prefix_name, **attributes)

        self.name = "BM25_with_"+self.tokenizer.name

    def is_trained(self):
        # use elastic search API instead to query the m_instance
        return exists(join(self.cache_folder, self.name))

    def create_index(self, corpora):

        # overide pre existent index. TODO change this behaviour
        self.es.indices.delete(index=self.prefix_name, ignore=[400, 404])

        # index structure
        self.es.indices.create(index=self.prefix_name,
                               body={"mappings": {
                                         "dynamic": "false",
                                         "properties": {
                                             "id": {
                                                 "type": "keyword"
                                                 },
                                             "text": {
                                                 "analyzer": "whitespace",
                                                 "type": "text"
                                                 },
                                             "original": {
                                                 "type": "keyword",
                                                 "store": "true"
                                                 },
                                             "title": {
                                                 "type": "keyword",
                                                 "store": "true"
                                                 }
                                             }
                                         }
                                     })

        def data_to_index_generator():
            # TODO: An improvement can be achived by looking at how elasticsearch use custom tokenizers
            index = 0
            for articles in corpora.read_documents_iterator():

                # batch tokenize following Keras CODE
                tokenized_articles = self.tokenizer.texts_to_sequences(map(lambda x: x["title"]+" "+x["abstract"], articles))

                for i in range(len(articles)):
                    yield {
                      "_index": self.prefix_name,
                      "id": articles[i]["id"],
                      "text": " ".join(map(lambda x: str(x), tokenized_articles[i])),
                      "original": articles[i]["title"]+" "+articles[i]["abstract"],
                      "title": articles[i]["title"]
                    }
                    index += 1
                    if not index % 100000:
                        log.info("{} documents indexed".format(index))

        helpers.bulk(self.es, data_to_index_generator(), chunk_size=1000, request_timeout=200)

        # save a empty file in cache to indicate that this index alredy exists. TODO use elasticsearch to do this.
        with open(join(self.cache_folder, self.name), "w") as f:
            json.dump({}, f)

    def retrieve_for_queries(self, query_data, name):
        """
        query_data: list {query:<str>, query_id:<int>, (optinal non relevant for this function) documents:<list - str>}
        """
        # TODO: Check elasticsearch for batch queries has a way of impriving
        retrieved_results = {}
        print("[BM25] Runing inference over data {}".format(name))
        for i, query_data in enumerate(query_data):
            query = ' '.join(map(lambda x: str(x), self.tokenizer.texts_to_sequences([query_data["query"]])[0]))
            query_es = {'query': {'bool': {'must': [{'query_string': {'query': query, 'analyze_wildcard': True}}], 'filter': [], 'should': [], 'must_not': []}}}

            retrieved = self.es.search(index=self.prefix_name, body=query_es, size=self.top_k)
            documents = list(map(lambda x: {"id": x['_source']['id'],
                                            "score": x['_score'],
                                            "original": x['_source']['original'],
                                            "title": x['_source']['title']},
                                 retrieved['hits']['hits']))
            log.info("[BM25] {}-query: {}".format(i, query_data["query_id"]))

            retrieved_results[query_data["query_id"]] = {"query": query_data["query"],
                                                         "documents": documents}

        return retrieved_results

    def train(self, **kwargs):
        steps = kwargs["steps"]
        corpora = kwargs["corpora"]
        queries = kwargs["queries"]
        model_output = {"origin": self.name,
                        "corpora": corpora,
                        "queries": queries,
                        "steps": steps}

        if "simulation" in kwargs:
            simulation = kwargs["simulation"]
        else:
            simulation = False

        # Start train routine
        if not self.tokenizer.is_trained():
            steps.append("[MISS] Tokenizer for the BM25")
            if not simulation:
                # code to create tokenizer
                print("[START] Create tokenizer for BM25")
                self.tokenizer.fit_tokenizer_multiprocess(corpora.read_documents_iterator(mapping=lambda x: x["title"]+" "+x["abstract"]))
                print("[FINISHED] tokenizer for BM25 with", len(self.tokenizer.word_counts), "terms")
        else:
            steps.append("[READY] Tokenizer for the BM25")

        if not self.is_trained():
            steps.append("[MISS] BM25 INDEX")
            if not simulation:
                # code to create index on elastic search
                self.create_index(corpora)
        else:
            steps.append("[READY] BM25 INDEX")

        # last step is to produce training output for the next module
        name = "{}_{}_{}_{}_retrieved_results.p".format(queries.train_name,
                                                        queries.validation_name,
                                                        self.name,
                                                        self.top_k)
        path_cache_output = join(self.cache_folder, name)
        if not exists(path_cache_output):
            steps.append("[MISS] BM25 INFERENCE")
            if not simulation:
                # run inference
                retrieved = self.inference(**kwargs)
                # save
                log.info("[BM25] Saving the retrieved documents in {}".format(path_cache_output))
                with open(path_cache_output, "wb") as f:
                    pickle.dump(retrieved, f)
        else:
            steps.append("[READY] BM25 INFERENCE")
            if not simulation:
                log.info("[BM25] Load the retrieved documents from {}".format(path_cache_output))
                with open(path_cache_output, "rb") as f:
                    retrieved = pickle.load(f)
                # rerun the show_evaluation
                if self.evaluation:
                    self.show_evaluation(retrieved["retrieved"], queries)

                # add to the output
                model_output["retrieved"] = retrieved["retrieved"]

        return model_output

    def __prepare_data(self, raw_predictions, raw_expectations):

        raw_predictions = dict(map(lambda x: (x[0], list(map(lambda k: k["id"], x[1]["documents"]))), raw_predictions.items()))
        raw_expectations = dict(map(lambda x: (x["query_id"], x["documents"]), raw_expectations))

        predictions = []
        expectations = []

        for _id in raw_expectations.keys():
            expectations.append(raw_expectations[_id])
            predictions.append(raw_predictions[_id])

        return predictions, expectations

    def show_evaluation(self, dict_results, queries):
        pred_train, expect_train = self.__prepare_data(dict_results["train"], queries.train_data)
        pred_validation, expect_validation = self.__prepare_data(dict_results["validation"], queries.validation_data)

        bioasq_map = "[BM25] BioASQ MAP@10: {}".format(f_map(pred_train, expect_train, bioASQ=True))
        print(bioasq_map)
        log.info(bioasq_map)
        map = "[BM25] Normal MAP@10: {}".format(f_map(pred_train, expect_train))
        print(map)
        log.info(map)
        recall = "[BM25] Normal RECALL@{}: {}".format(self.top_k, f_recall(pred_train, expect_train, at=self.top_k))
        print(recall)
        log.info(recall)

    def inference(self, **kwargs):
        steps = kwargs["steps"]
        model_output = {"origin": self.name,
                        "steps": steps}

        if "simulation" in kwargs:
            simulation = kwargs["simulation"]
        else:
            simulation = False

        if not self.is_trained():
            steps.append("[MISS/CRITICAL] BM25 TRAIN")
            if not simulation:
                raise Exception("BM25 is not trained")
        else:
            steps.append("[READY] BM25 INFERENCE")
            if not simulation:
                # code to infer over the queries
                if "queries" in kwargs:
                    queries = kwargs["queries"]
                    train_out = self.retrieve_for_queries(queries.train_data, "train")
                    validation_out = self.retrieve_for_queries(queries.validation_data, "validation")
                    model_output["retrieved"] = {"train": train_out, "validation": validation_out}

                    # perform evaluation
                    if self.evaluation:
                        self.show_evaluation(model_output["retrieved"], queries)
                elif "query" in kwargs:
                    query = kwargs["query"]
                    model_output["query"] = query
                    # RUN A SINGLE Query
                    model_output["query_out"] = self.retrieve_for_queries(query, "query")

        return model_output
