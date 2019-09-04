from models.model import ModelAPI
from utils import dynamicly_class_load
from os.path import exists, join


class BM25_ES(ModelAPI):
    def __init__(self, prefix_name, cache_folder, top_k, tokenizer):
        self.top_k = top_k
        self.cache_folder = cache_folder
        self.prefix_name = prefix_name.name

        name, attributes = list(tokenizer.items())[0]
        _class = dynamicly_class_load("tokenizers."+name, name)
        self.tokenizer = _class.maybe_load(cache_folder=cache_folder, prefix_name=self.prefix_name, **attributes)

        self.name = "BM25_with_"+self.tokenizer.name

    def is_trained(self):
        # use elastic search API instead to query the m_instance
        return exists(join(self.cache_folder, self.name))

    def train(self, simulation=False, **kwargs):
        steps = []
        model_output = None

        if "corpora" in kwargs:
            corpora = kwargs.pop("corpora")

        if "queries" in kwargs:
            queries = kwargs.pop("queries")

        if kwargs:
            raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))

        # Start train routine
        if not self.tokenizer.is_trained():
            steps.append("[MISS] Tokenizer for the BM25")
            if not simulation:
                # code to create tokenizer
                print("[START] Create tokenizer for BM25")
                self.tokenizer.fit_tokenizer_multiprocess(corpora.read_documents_generator())
                print("[FINISHED] tokenizer for BM25 with", self.tokenizer.num_words, "terms")

        else:
            steps.append("[READY] Tokenizer for the BM25")

        if not self.is_trained():
            steps.append("[MISS] BM25 INDEX")
            if not simulation:
                # code to create index
                raise NotImplementedError()
        else:
            steps.append("[READY] BM25 INDEX")

        if not self.is_trained():  # this condition only makes sence if simulation == True or if something went wrong
            steps.append("[MISS] BM25 INFERENCE")
        else:
            steps.append("[READY] BM25 INFERENCE")
            if not simulation:
                # code to infer over the queries
                raise NotImplementedError()

        if simulation:
            return steps
        else:
            return model_output
