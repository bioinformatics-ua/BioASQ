from tokenizers.base_tokenizer import BaseTokenizer
from bllipparser import tokenize as bllip_tokenize
from nltk.stem import PorterStemmer
import string
import sys
from os.path import exists, join
import json

if sys.version_info < (3,):
    maketrans = string.maketrans
else:
    maketrans = str.maketrans


class Bllip(BaseTokenizer):
    def __init__(self, stem=False, **kwargs):
        super().__init__(**kwargs)
        self.st = PorterStemmer() if stem else None
        self.name = self.prefix_name + "_" + ("stem_" if stem else "")+"Bllip"

    def get_properties(self):
        return {"cache_folder": self.cache_folder, "prefix_name": self.prefix_name, "stem": self.stem}

    def tokenizer(self, text, *args, **kwargs):
        tokens = []
        text = text.lower()
        sentences = text.split(".")
        filters = '!"#$%&*,/:;()[]{}<=>?@\\^_`|~'
        tab = maketrans(filters, " "*(len(filters)))

        for i, sentence in enumerate(sentences):
            if i == len(sentences)-1:
                tokens.extend(bllip_tokenize(sentence.translate(tab)))
            else:
                tokens.extend(bllip_tokenize((sentence+".").translate(tab)))

        if self.st is not None:
            tokens = [self.st.stem(token) for token in tokens]

        return tokens

    def get_config(self):
        t_config = super().get_config()
        t_config["stem"] = json.dumps(self.stem)
        return t_config

    def save_to_json(self, **kwargs):
        """
        KERAS code

        Returns a JSON string containing the tokenizer configuration.
        To load a tokenizer from a JSON string, use
        `keras.preprocessing.text.tokenizer_from_json(json_string)`.

        # Returns
            A JSON string containing the tokenizer configuration.
        """
        if 'path' in kwargs:
            path = kwargs.pop('path')
        else:
            path = join(self.cache_folder, self.name+".json")
        with open(path, "w") as f:
            json.dump(self.get_config(), f)

    @staticmethod
    def load_from_json(path):
        with open(path, "r") as f:
            t_config = json.load(f)
            return Bllip(**t_config)

    @staticmethod
    def maybe_load(cache_folder, prefix_name, stem, **kwargs):

        # prefix_name and stem should be in the kwargs
        name = prefix_name + "_" + ("stem_" if stem else "")+"Bllip.json"
        path = join(cache_folder, name)
        if exists(path):
            print("[LOAD FROM CACHE] Load regex tokenizer from", path)
            return Bllip.load_from_json(path)

        return Bllip(stem, cache_folder=cache_folder, prefix_name=prefix_name, **kwargs)
