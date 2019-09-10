import tarfile
import codecs
import json
import gc
import os
from logger import log


class Corpora:
    def __init__(self, name, folder, files_are_compressed=False):
        self.name = name
        self.folder_rep = folder
        self.files_are_compressed = files_are_compressed

    def read_documents_iterator(self, mapping=None):
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
                    log.info("[CORPORA] Openning tar file {}".format(m.name))
                    f = tar.extractfile(m)
                    articles = json.load(reader(f))
                    log.info("[CORPORA] Returning: articles {}".format(len(articles)))
                    if mapping is not None:
                        articles = list(map(mapping, articles))
                    yield articles
                    f.close()
                    del f
                    log.info("[CORPORA] Force garbage collector ({})".format(gc.collect()))

            return generator()
        else:
            raise NotImplementedError("In the current version the documents must be in json format and compressed (tar.gz) ")


class Queries:
    def __init__(self, mode, train_file, validation_file):
        self.train_file = train_file
        self.validation_file = validation_file
        self.train_name = os.path.basename(self.train_file).split(".")[0]
        self.validation_name = os.path.basename(self.validation_file).split(".")[0]

        self.mode = mode

        self.train_data = []
        self.train_data_dict = {}
        self.positive_ids = []
        self.validation_data = []
        self.validation_data_dict = {}

        self.__load()

    def __load(self):
        with open(self.train_file, "r") as f:
            self.train_data.extend(json.load(f))

        self.positive_ids = []
        for data in self.train_data:
            self.positive_ids.extend(data["documents"])
        self.positive_ids = set(self.positive_ids)

        with open(self.validation_file, "r") as f:
            self.validation_data.extend(json.load(f))

        self.train_data_dict = dict(map(lambda x: (x["query_id"], {"documents": x["documents"], "query": x["query"]}), self.train_data))
        self.validation_data_dict = dict(map(lambda x: (x["query_id"], {"documents": x["documents"], "query": x["query"]}), self.validation_data_dict))
