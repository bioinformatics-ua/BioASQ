from models.model import ModelAPI
from utils import dynamicly_class_load
from os.path import exists, join


class DeepRank(ModelAPI):
    def __init__(self, prefix_name, cache_folder, logging, top_k, tokenizer, **kwargs):
        self.top_k = top_k
        self.cache_folder = cache_folder
        self.prefix_name = prefix_name
        self.logging = logging

        name, attributes = list(tokenizer.items())[0]
        _class = dynamicly_class_load("tokenizers."+name, name)
        self.tokenizer = _class.maybe_load(cache_folder=cache_folder, prefix_name=self.prefix_name, **attributes)

        self.name = "DeepRank_with_"+self.tokenizer.name

    def is_trained(self):
        # use elastic search API instead to query the m_instance
        return exists(join(self.cache_folder, self.name))

    def train(self, simulation=False, **kwargs):
        steps = []
        model_output = None

        if not self.tokenizer.is_trained():
            steps.append("[MISS] Tokenizer for the DeepRank")
            if not simulation:
                # code to create tokenizer
                raise NotImplementedError()
        else:
            steps.append("[READY] Tokenizer for the DeepRank")

        if not self.is_trained():
            steps.append("[MISS] DeepRank TRAIN")
            if not simulation:
                # code to create index
                raise NotImplementedError()
        else:
            steps.append("[READY] DeepRank TRAIN")

        if simulation:
            return steps
        else:
            return model_output

    def inference(self, simulation=False, **kwargs):
        return []
