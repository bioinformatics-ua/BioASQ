import tarfile
import codecs
import json
import gc
from os.path import join


class Corpora:
    def __init__(self, name, folder, logging, files_are_compressed=False):
        self.name = name
        self.folder_rep = folder
        self.logging = logging
        self.files_are_compressed = files_are_compressed

    def read_documents_generator(self, mapping=None):
        """
        creates a generator to read the document collection
        """
        if self.files_are_compressed:
            reader = codecs.getreader("ascii")
            tar = tarfile.open(self.folder_rep)

            print("[CORPORA] Openning tar file", self.folder_rep)

            members = tar.getmembers()

            def generator():
                for m in members:
                    self.logging.info("[CORPORA] Openning tar file", m.name)
                    f = tar.extractfile(m)
                    articles = json.load(reader(f))
                    self.logging.info("[CORPORA] Returning:", len(articles), "articles")
                    if mapping is not None:
                        articles = list(map(mapping, articles))
                    yield articles
                    f.close()
                    del f
                    self.logging.info("[CORPORA] Force garbage collector", gc.collect())

            return generator
        else:
            raise NotImplementedError("In the current version the documents must be in json format and compressed (tar.gz) ")


class Queries:
    def __init__(self, mode, folder):
        self.folder_rep = folder
        self.mode = mode

        self.train_data = []
        self.validation_data = []

        self.__load()

    def __load(self):
        with open(join(self.folder_rep, "train.json"), "r") as f:
            self.train_data.extend(json.load(f))

        with open(join(self.folder_rep, "validation.json"), "r") as f:
            self.validation_data.extend(json.load(f))
