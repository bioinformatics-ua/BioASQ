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
    def __init__(self, stem=False, sw_file=None, queries_sw=False, articles_sw=False, **kwargs):
        super().__init__(**kwargs)
        if isinstance(stem, str):
            self.stem = stem == "true"
        else:
            self.stem = stem
        self.st = PorterStemmer() if stem else None
        self.sw_file = sw_file

        if isinstance(queries_sw, str):
            self.queries_sw = queries_sw == "true"
        else:
            self.queries_sw = queries_sw

        if isinstance(articles_sw, str):
            self.articles_sw = articles_sw == "true"
        else:
            self.articles_sw = articles_sw

        self.pattern = re.compile('[^a-zA-Z0-9\s]+')
        self.filter_whitespace = lambda x: not x == ""
        self.name = self.prefix_name + "_" + ("stem_" if self.stem else "")+"Regex"
        self.name_properties = self.prefix_name + "_" + ("stem_" if self.stem else "")+"Regex_"+str(self.queries_sw)+"_"+str(self.articles_sw)
        print("DEBUG created tokenizer", self.name)
        if self.sw_file is not None:
            with open(self.sw_file, "r") as f:
                self.stop_words = json.load(f)
        self.stop_words_tokenized = None

        print(self.queries_sw, self.articles_sw)

    def get_properties(self):
        return {"cache_folder": self.cache_folder,
                "prefix_name": self.prefix_name,
                "stem": self.stem,
                "queries_sw": self.queries_sw,
                "articles_sw": self.articles_sw,
                "sw_file": self.sw_file}

    def tokenize_texts(self, texts, **kwargs):

        if kwargs["mode"] == "queries":
            flag = self.queries_sw
        elif kwargs["mode"] == "articles":
            flag = self.articles_sw

        tokenized_texts = self.texts_to_sequences(texts)
        if flag:
            if self.stop_words_tokenized is None:  # lazzy initialization
                self.stop_words_tokenized = set(self.texts_to_sequences([self.stop_words])[0])

            for tokenized_text in tokenized_texts:
                tokenized_text = [token for token in tokenized_text if token not in self.stop_words_tokenized]

        return tokenized_texts

    def tokenize_query(self, query):
        tokenized_query = self.texts_to_sequences([query])[0]
        if self.queries_sw:
            if self.stop_words_tokenized is None:  # lazzy initialization
                self.stop_words_tokenized = set(self.texts_to_sequences([self.stop_words])[0])
            tokenized_query = [token for token in tokenized_query if token not in self.stop_words_tokenized]

        return tokenized_query

    def tokenize_article(self, article):
        tokenized_article = self.texts_to_sequences([article])[0]
        if self.articles_sw:
            if self.stop_words_tokenized is None:  # lazzy initialization
                self.stop_words_tokenized = set(self.texts_to_sequences([self.stop_words])[0])
            tokenized_article = [token for token in tokenized_article if token not in self.stop_words_tokenized]

        return tokenized_article

    def tokenizer(self, text, *args, **kwargs):
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
        t_config["stem"] = self.stem
        if self.sw_file is not None:
            t_config["sw_file"] = self.sw_file
            t_config["queries_sw"] = self.queries_sw
            t_config["articles_sw"] = self.articles_sw
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
    def load_from_json(path, **kwargs):
        with open(path, "r") as f:
            t_config = json.load(f)
            # override - TODO change this is stupid this way
            t_config["queries_sw"] = kwargs["queries_sw"]
            t_config["articles_sw"] = kwargs["articles_sw"]
            t_config["sw_file"] = kwargs["sw_file"]
            return Regex(**t_config)

    @staticmethod
    def maybe_load(cache_folder, prefix_name, stem, **kwargs):

        # prefix_name and stem should be in the kwargs
        name = prefix_name + "_" + ("stem_" if stem else "")+"Regex.json"
        path = join(cache_folder, name)
        if exists(path):
            print("[LOAD FROM CACHE] Load tokenizer from", path)
            return Regex.load_from_json(path, **kwargs)

        return Regex(stem, cache_folder=cache_folder, prefix_name=prefix_name, **kwargs)
