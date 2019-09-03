from elasticsearch import Elasticsearch, helpers

from models.generic_model import ModelAPI
from pubmed_data import pubmed_helper as ph

import time

class BM25_ES(ModelAPI):

    def __init__(self, saved_models_path=None):
        if saved_models_path is None:
            super().__init__()
        else:
            super().__init__(saved_models_path=saved_models_path)

        self.es = Elasticsearch(['http://193.136.175.98:8125'])
        self.tokenizer_mode = "bllip_stem_full_tokens"
        self.tk = ph.load_tokenizer(mode = self.tokenizer_mode)
        self.trained = True
        
    def _training_process(self, data):
        
        gen = ph.create_tokenized_pubmed_collection_generator(mode = self.tokenizer_mode)
        
        pmid_index_map = ph.pmid_index_mapping()
        
        self.es.indices.delete(index='bioasq', ignore=[400, 404])

        self.es.indices.create(
          index="bioasq",
          body={
            "mappings": {
              "dynamic": "false",
              "properties": {
                "pmid": {
                  "type": "keyword"
                },
                "text": {
                  "analyzer":"whitespace",
                  "type": "text"
                }
              }
            }
          }
        )

        def readdata():

            index = 0

            for articles in gen():
                for article in articles:
                    #try:
                    yield {
                      "_index": "bioasq",
                      "pmid": pmid_index_map.inverse[index],
                      "text": " ".join(map(lambda x: str(x), article))
                    }
                    #except Exception as e:
                    #    print(e)
                    index+=1

                    if not index%10000: print("%i terms processed..." %index)

        helpers.bulk(es, readdata(), chunk_size=1000, request_timeout=200)

            
    
    
    def _predict_process(self, queries, **kwargs):
        """
        queries - List of queries, here each is represented by json {id:<id>, body:<string>}
        
        """
        
        if 'top_k' in kwargs:
            top_k = kwargs.pop('top_k')
        else:
            top_k = 10000
        
        if kwargs:
            raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))

        results = {}
        
        for i,query_data in enumerate(queries):
            print("Query",i,end="\r")
            results[query_data["id"]] =  {"body":query_data["body"],
                                          "documents":self.__single_query(query_data,top_k)}
        
        return results
    
    def __single_query(self, query_data, top_k):
        
        strat_t = time.time()
        
        query = ' '.join(map(lambda x:str(x), self.tk.texts_to_sequences([query_data["body"]])[0]))

        # try simple query
        q = {'query': {'bool': {'must': [{'query_string': {'query': '', 'analyze_wildcard': True}}], 'filter': [], 'should': [], 'must_not': []}}}

        # change this field for each query
        q['query']['bool']['must'][0]['query_string']['query'] = query

        res = self.es.search(index="bioasq", body=q, size=top_k, request_timeout=200)
        
        output_format = list(map(lambda x:(x['_source']['pmid'],x['_score'],x['_source']['original'],x['_source']['title']),res['hits']['hits']))
        
        print("BM25 ES time", time.time() - strat_t)
        
        return output_format
        

        
    @staticmethod
    def load(path = '/backup/saved_models/',full_tokens = False):
        pass
    
    def save(self):
        pass
