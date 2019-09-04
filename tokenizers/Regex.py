from nltk.stem import PorterStemmer
from tokenizers.base_tokenizer import BaseTokenizer
import re
import string
import sys
import json
from os.path import exists, join


if sys.version_info < (3,):
    maketrans = string.maketrans
else:
    maketrans = str.maketrans


class Regex(BaseTokenizer):
    def __init__(self, stem=False, **kwargs):
        super().__init__(**kwargs)
        self.stem = stem
        self.st = PorterStemmer() if stem else None
        self.pattern = re.compile('[^a-zA-Z0-9\s]+')
        self.filter_whitespace = lambda x: not x == ""
        self.name = self.prefix_name + "_" + ("stem_" if stem else "")+"Regex"

    def tokenizer(self, text):
        text = text.lower()
        text = (text.encode("ascii", "replace").decode("utf-8"))
        filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        tab = maketrans(filters, " "*(len(filters)))
        text = text.translate(tab)
        tokens = self.pattern.sub('', text).split(" ")
        tokens = list(filter(self.filter_whitespace, tokens))

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
            return Regex(**t_config)

    @staticmethod
    def maybe_load(cache_folder, prefix_name, stem, **kwargs):

        # prefix_name and stem should be in the kwargs
        name = prefix_name + "_" + ("stem_" if stem else "")+"Regex.json"
        path = join(cache_folder, name)
        if exists(path):
            print("[LOAD FROM CACHE] Load regex tokenizer from", path)
            return Regex.load_from_json(path)

        return Regex(stem, cache_folder=cache_folder, prefix_name=prefix_name)
