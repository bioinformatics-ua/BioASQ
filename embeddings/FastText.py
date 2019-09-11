from os.path import exists, join, basename
import fasttext
import pickle
from logger import log
import gc
import numpy as np


class FastText:
    def __init__(self, cache_folder, prefix_name, tokenizer, path, trainable):
        self.cache_folder = cache_folder
        self.prefix_name = prefix_name
        self.path = path
        self.trainable = trainable
        self.tokenizer = tokenizer
        self.vocab_size = None
        self.embedding_size = None
        self.matrix = None
        self.name = FastText.get_name(self.path, self.tokenizer.name)

    @staticmethod
    def get_name(file_path, tokenizer_name):
        return "embedding_{}_{}".format(basename(file_path).split(".")[0],
                                        tokenizer_name)

    @staticmethod
    def maybe_load(cache_folder, prefix_name, tokenizer, path, trainable):
        name = FastText.get_name(path, tokenizer.name)
        cache_path = join(cache_folder, name)
        if exists(cache_path):
            print("[LOAD FROM CACHE] Load embedding matrix from", cache_path)
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
                ft = FastText(cache_folder, prefix_name, tokenizer, path, trainable)
                ft.matrix = data["matrix"]
                ft.vocab_size = data["vocab_size"]
                ft.embedding_size = data["embedding_size"]
                return ft
        else:
            return FastText(cache_folder, prefix_name, tokenizer, path, trainable)

    def has_matrix(self):
        return self.matrix is not None

    def build_matrix(self):
        print("[FASTTEXT] Creating embedding matrix")
        trained_ft_model = None
        try:
            log.info("[FASTTEXT] load model")
            trained_ft_model = fasttext.load_model(self.path)
            log.info("[FASTTEXT] build embedding matrix")
            self.vocab_size = self.tokenizer.vocabulary_size()
            self.embedding_size = trained_ft_model.get_dimension()

            self.matrix = np.zeros((self.vocab_size, self.embedding_size))

            for index, word in self.tokenizer.index_word.items():
                _temp = trained_ft_model[word]
                # normalize
                self.matrix[index] = _temp/np.linalg.norm(_temp)
        except Exception as e:
            log.error(e)
            raise e
        finally:
            del trained_ft_model
            log.info("GC ({})".format(gc.collect()))

        # save after gc
        cache_path = join(self.cache_folder, self.name)
        with open(cache_path, "wb") as f:
            log.info("[FASTTEXT] save")
            data = {"vocab_size": self.vocab_size,
                    "embedding_size": self.embedding_size,
                    "matrix": self.matrix}
            pickle.dump(data, f, protocol=4)

    def embedding_matrix(self):
        if not self.has_matrix():
            if not self.tokenizer.is_trained():
                raise "At this point the tokenizer should already have been trained"
            else:
                # build matrix and save
                self.build_matrix()

        return self.matrix
