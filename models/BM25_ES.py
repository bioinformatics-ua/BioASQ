from models.model import ModelAPI
from utils import dynamicly_class_load
from os.path import exists, join
from elasticsearch import Elasticsearch, helpers
import json


class BM25_ES(ModelAPI):
    def __init__(self, prefix_name, cache_folder, logging, top_k, address, tokenizer):
        self.top_k = top_k
        self.cache_folder = cache_folder
        self.prefix_name = prefix_name
        self.logging = logging
        self.es = Elasticsearch([address])

        name, attributes = list(tokenizer.items())[0]
        _class = dynamicly_class_load("tokenizers."+name, name)
        self.tokenizer = _class.maybe_load(cache_folder=cache_folder, prefix_name=self.prefix_name, **attributes)

        self.name = "BM25_with_"+self.tokenizer.name

    def is_trained(self):
        # use elastic search API instead to query the m_instance
        return exists(join(self.cache_folder, self.name))

    def create_index(self, corpora):
        exit(1)
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
            for articles in corpora.read_documents_generator():

                # batch tokenize following Keras CODE
                tokenized_articles = self.tokenizer.texts_to_sequences(map(lambda x: x["title"]+" "+x["abstract"], articles))

                for i in len(articles):
                    yield {
                      "_index": "bioasq",
                      "id": articles[i]["id"],
                      "text": " ".join(map(lambda x: str(x), tokenized_articles[i])),
                      "original": articles[i]["title"]+" "+articles[i]["abstract"],
                      "title": articles[i]["title"]
                    }
                    index += 1
                    if not index % 100000:
                        self.logging.info("{} documents indexed".format(index))

        helpers.bulk(self.es, data_to_index_generator(), chunk_size=1000, request_timeout=200)

        # save a empty file in cache to indicate that this index alredy exists. TODO use elasticsearch to do this.
        with open(join(self.cache_folder, self.name)) as f:
            json.dump({}, f)

    def retrieve_for_queries(self, queries):
        # TODO: Check elasticsearch for batch queries has a way of impriving
        retrieved_results = []
        for query_data in queries.train:
            query = ' '.join(map(lambda x: str(x), self.tokenizer.texts_to_sequences([query_data["query"]])[0]))
            query_es = {'query': {'bool': {'must': [{'query_string': {'query': query, 'analyze_wildcard': True}}], 'filter': [], 'should': [], 'must_not': []}}}

            retrieved = self.es.search(index=self.prefix_name, body=query_es, size=self.top_k)
            documents = list(map(lambda x: {"id": x['_source']['id'],
                                            "score": x['_score'],
                                            "original": x['_source']['original'],
                                            "title": x['_source']['title']},
                                 retrieved['hits']['hits']))

            retrieved_results.append({"query_id": query_data["query_id"],
                                      "query": query_data["query"],
                                      "documents": documents})

        return retrieved_results

    def train(self, simulation=False, **kwargs):
        steps = []

        if "corpora" in kwargs:
            corpora = kwargs.pop("corpora")

        # Start train routine
        if not self.tokenizer.is_trained():
            steps.append("[MISS] Tokenizer for the BM25")
            if not simulation:
                # code to create tokenizer
                print("[START] Create tokenizer for BM25")
                self.tokenizer.fit_tokenizer_multiprocess(corpora.read_documents_generator(mapping=lambda x: x["title"]+" "+x["abstract"]))
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

        return steps

    def inference(self, simulation=False, **kwargs):
        steps = []
        model_output = None

        if "queries" in kwargs:
            queries = kwargs.pop("queries")

        if not self.is_trained():
            steps.append("[MISS] BM25 TRAIN")
        else:
            steps.append("[READY] BM25 INFERENCE")
            if not simulation:
                # code to infer over the queries
                model_output = self.retrieve_for_queries(queries)

        if simulation:
            return steps
        else:
            return model_output
