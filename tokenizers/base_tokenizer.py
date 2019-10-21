"""
This class is a simplification of the tensorflow keras tokenizer with some changes
But the original source code can be found on: https://www.tensorflow.org/versions/r1.9/api_docs/python/tf/keras/preprocessing/text/Tokenizer
"""

from collections import OrderedDict
from collections import defaultdict
import tempfile
import pickle
import shutil
import json
import os
from multiprocessing import Process
import gc
import sys
from logger import log


def fitTokenizeJob(proc_id, articles, _class, merge_tokenizer_path, properties):
    print("[Process-{}] Started".format(proc_id))
    sys.stdout.flush()
    # ALL THREADS RUN THIS
    tk = _class(**properties)
    tk.fit_on_texts(articles)
    del articles

    file_name = "tk_{0:03}.p".format(proc_id)
    print("[Process-{}]: Store {}".format(proc_id, file_name))

    tk.save_to_json(path=os.path.join(merge_tokenizer_path, file_name))
    del tk
    print("[Process-{}] Ended".format(proc_id))


def tokenizeJob(proc_id, texts, _class, merge_tokenizer_path, properties, kwargs):
    print("[Process-{}] Started articles size {}".format(proc_id, len(texts)))

    sys.stdout.flush()
    # load tokenizer
    tokenizer = _class.maybe_load(**properties)

    # ALL THREADS RUN THIS
    tokenized_texts = tokenizer.tokenize_texts(texts, **kwargs)
    del texts

    file_name = "tokenized_text_{0:03}.p".format(proc_id)
    print("[Process-{}]: Store {}".format(proc_id, file_name))

    with open(os.path.join(merge_tokenizer_path, file_name), "wb") as f:
        pickle.dump(tokenized_texts, f)

    del tokenized_texts
    gc.collect()

    print("[Process-{}] Ended".format(proc_id))


class BaseTokenizer:
    """Text tokenization utility class.

    This class allows to vectorize a text corpus, by turning each
    text into either a sequence of integers (each integer being the index
    of a token in a dictionary)

    # Arguments
        num_words: the maximum number of words to keep, based
            on word frequency. Only the most common `num_words-1` words will
            be kept.
        filters: a string where each element is a character that will be
            filtered from the texts. The default is all punctuation, plus
            tabs and line breaks, minus the `'` character.
        lower: boolean. Whether to convert the texts to lowercase.
        split: str. Separator for word splitting.
        char_level: if True, every character will be treated as a token.
        oov_token: if given, it will be added to word_index and used to
            replace out-of-vocabulary words during text_to_sequence calls

    `0` is a reserved index that won't be assigned to any word.
    """

    def __init__(self, cache_folder,
                 prefix_name,
                 num_words=None,
                 filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                 lower=True,
                 split=' ',
                 char_level=False,
                 oov_token=None,
                 document_count=0,
                 n_process=4,
                 **kwargs):

        self.prefix_name = prefix_name
        self.cache_folder = cache_folder
        if 'word_counts' in kwargs:
            self.word_counts = json.loads(kwargs.pop('word_counts'))
        else:
            self.word_counts = OrderedDict()
        if 'word_docs' in kwargs:
            self.word_docs = json.loads(kwargs.pop('word_docs'))
        else:
            self.word_docs = defaultdict(int)
        self.filters = filters
        self.split = split
        self.lower = lower
        self.num_words = num_words
        self.document_count = document_count
        self.char_level = char_level
        self.oov_token = oov_token
        if 'index_docs' in kwargs:
            self.index_docs = json.loads(kwargs.pop('index_docs'))
            self.index_docs = {int(k): v for k, v in self.index_docs.items()}
        else:
            self.index_docs = defaultdict(int)
        if 'word_index' in kwargs:
            self.word_index = json.loads(kwargs.pop('word_index'))
        else:
            self.word_index = dict()
        if 'index_word' in kwargs:
            self.index_word = json.loads(kwargs.pop('index_word'))
            self.index_word = {int(k): v for k, v in self.index_word.items()}
        else:
            self.index_word = dict()
        self.n_process = n_process

        if kwargs:
            raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))

    def tokenizer(self, text, *args, **kwargs):
        raise NotImplementedError("The function tokenizer must be implemented by a subclass")

    @staticmethod
    def maybe_load(path, **kwargs):
        raise NotImplementedError("The function tokenizer must be implemented by a subclass")

    def fit_on_texts(self, texts):
        """Updates internal vocabulary based on a list of texts.

        In the case where texts contains lists,
        we assume each entry of the lists to be a token.

        Required before using `texts_to_sequences` or `texts_to_matrix`.

        # Arguments
            texts: can be a list of strings,
                a generator of strings (for memory-efficiency),
                or a list of list of strings.
        """
        for text in texts:
            self.document_count += 1
            if self.char_level or isinstance(text, list):
                if self.lower:
                    if isinstance(text, list):
                        text = [text_elem.lower() for text_elem in text]
                    else:
                        text = text.lower()
                seq = text
            else:
                seq = self.tokenizer(text)
            for w in seq:
                if w in self.word_counts:
                    self.word_counts[w] += 1
                else:
                    self.word_counts[w] = 1
            for w in set(seq):
                # In how many documents each word occurs
                self.word_docs[w] += 1

        wcounts = list(self.word_counts.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)
        # forcing the oov_token to index 1 if it exists
        if self.oov_token is None:
            sorted_voc = []
        else:
            sorted_voc = [self.oov_token]
        sorted_voc.extend(wc[0] for wc in wcounts)

        # note that index 0 is reserved, never assigned to an existing word
        self.word_index = dict(
            list(zip(sorted_voc, list(range(1, len(sorted_voc) + 1)))))

        self.index_word = dict((c, w) for w, c in self.word_index.items())

        for w, c in list(self.word_docs.items()):
            self.index_docs[self.word_index[w]] = c

    def texts_to_sequences(self, texts):
        """Transforms each text in texts to a sequence of integers.

        Only top `num_words-1` most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.

        # Arguments
            texts: A list of texts (strings).

        # Returns
            A list of sequences.
        """
        return list(self.texts_to_sequences_generator(texts))

    def texts_to_sequences_generator(self, texts):
        """Transforms each text in `texts` to a sequence of integers.

        Each item in texts can also be a list,
        in which case we assume each item of that list to be a token.

        Only top `num_words-1` most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.

        # Arguments
            texts: A list of texts (strings).

        # Yields
            Yields individual sequences.
        """
        num_words = self.num_words
        oov_token_index = self.word_index.get(self.oov_token)
        for text in texts:
            if self.char_level or isinstance(text, list):
                if self.lower:
                    if isinstance(text, list):
                        text = [text_elem.lower() for text_elem in text]
                    else:
                        text = text.lower()
                seq = text
            else:
                seq = self.tokenizer(text,
                                     self.filters,
                                     self.lower,
                                     self.split)

            vect = []
            for w in seq:
                i = self.word_index.get(w)
                if i is not None:
                    if num_words and i >= num_words:
                        if oov_token_index is not None:
                            vect.append(oov_token_index)
                    else:
                        vect.append(i)
                elif self.oov_token is not None:
                    vect.append(oov_token_index)
            yield vect

    def sequences_to_texts(self, sequences):
        """Transforms each sequence into a list of text.

        Only top `num_words-1` most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.

        # Arguments
            sequences: A list of sequences (list of integers).

        # Returns
            A list of texts (strings)
        """
        return list(self.sequences_to_texts_generator(sequences))

    def sequences_to_texts_generator(self, sequences):
        """Transforms each sequence in `sequences` to a list of texts(strings).

        Each sequence has to a list of integers.
        In other words, sequences should be a list of sequences

        Only top `num_words-1` most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.

        # Arguments
            sequences: A list of sequences.

        # Yields
            Yields individual texts.
        """
        num_words = self.num_words
        oov_token_index = self.word_index.get(self.oov_token)
        for seq in sequences:
            vect = []
            for num in seq:
                word = self.index_word.get(num)
                if word is not None:
                    if num_words and num >= num_words:
                        if oov_token_index is not None:
                            vect.append(self.index_word[oov_token_index])
                    else:
                        vect.append(word)
                elif self.oov_token is not None:
                    vect.append(self.index_word[oov_token_index])
            vect = ' '.join(vect)
            yield vect

    def is_trained(self):
        return len(self.word_counts) > 0

    def vocabulary_size(self):
        return len(self.index_word) + 1  # because 0 index is reserved

    def get_config(self):
        '''Returns the tokenizer configuration as Python dictionary.
        The word count dictionaries used by the tokenizer get serialized
        into plain JSON, so that the configuration can be read by other
        projects.

        # Returns
            A Python dictionary with the tokenizer configuration.
        '''
        json_word_counts = json.dumps(self.word_counts)
        json_word_docs = json.dumps(self.word_docs)
        json_index_docs = json.dumps(self.index_docs)
        json_word_index = json.dumps(self.word_index)
        json_index_word = json.dumps(self.index_word)

        return {
            'cache_folder': self.cache_folder,
            'prefix_name': self.prefix_name,
            'num_words': self.num_words,
            'filters': self.filters,
            'lower': self.lower,
            'split': self.split,
            'char_level': self.char_level,
            'oov_token': self.oov_token,
            'document_count': self.document_count,
            'word_counts': json_word_counts,
            'word_docs': json_word_docs,
            'index_docs': json_index_docs,
            'index_word': json_index_word,
            'word_index': json_word_index
        }

    def save_to_json(self, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def load_from_json(path):
        raise NotImplementedError()

    def get_properties(self):
        raise NotImplementedError()

    def fit_tokenizer_multiprocess(self, corpora_iterator):
        merge_tokenizer_path = tempfile.mkdtemp()

        try:
            # initialization of the process
            def fitTokenizer_process_init(proc_id, articles):
                return Process(target=fitTokenizeJob, args=(proc_id, articles, self.__class__, merge_tokenizer_path, self.get_properties()))

            # multiprocess loop
            for i, texts in enumerate(corpora_iterator):
                process = []

                t_len = len(texts)
                t_itter = t_len//self.n_process

                for k, j in enumerate(range(0, t_len, t_itter)):
                    process.append(fitTokenizer_process_init(self.n_process*i+k, texts[j:j+t_itter]))

                print("[MULTIPROCESS LOOP] Starting", self.n_process, "process")
                for p in process:
                    p.start()

                print("[MULTIPROCESS LOOP] Wait", self.n_process, "process")
                for p in process:
                    p.join()
                gc.collect()

            # merge the tokenizer
            print("[TOKENIZER] Merge")
            files = sorted(os.listdir(merge_tokenizer_path))

            for file in files:
                log.info("[TOKENIZER] Load {}".format(file))
                loaded_tk = self.__class__.load_from_json(path=os.path.join(merge_tokenizer_path, file), **self.get_properties())

                # manual merge
                for w, c in loaded_tk.word_counts.items():
                    if w in self.word_counts:
                        self.word_counts[w] += c
                    else:
                        self.word_counts[w] = c

                for w, c in loaded_tk.word_docs.items():
                    if w in self.word_docs:
                        self.word_docs[w] += c
                    else:
                        self.word_docs[w] = c

                self.document_count += loaded_tk.document_count

                # CODE FROM KERAS TOKENIZER
                wcounts = list(self.word_counts.items())
                wcounts.sort(key=lambda x: x[1], reverse=True)
                # forcing the oov_token to index 1 if it exists
                if self.oov_token is None:
                    sorted_voc = []
                else:
                    sorted_voc = [self.oov_token]
                sorted_voc.extend(wc[0] for wc in wcounts)

                # note that index 0 is reserved, never assigned to an existing word
                self.word_index = dict(
                    list(zip(sorted_voc, list(range(1, len(sorted_voc) + 1)))))

                self.index_word = dict((c, w) for w, c in self.word_index.items())

                for w, c in list(self.word_docs.items()):
                    self.index_docs[self.word_index[w]] = c

                # Saving tokenizer
                self.save_to_json()
        except Exception as e:
            raise e
        finally:
            # always remove the temp directory
            log.info("[TOKENIZER] Remove {}".format(merge_tokenizer_path))
            shutil.rmtree(merge_tokenizer_path)

    def tokenizer_multiprocess(self, texts, **kwargs):
        merge_tokenizer_path = tempfile.mkdtemp()
        tokenized_texts = []

        try:
            # initialization of the process
            def tokenizer_process_init(proc_id, texts):
                return Process(target=tokenizeJob, args=(proc_id, texts, self.__class__, merge_tokenizer_path, self.get_properties(), kwargs))

            # multiprocess loop
            itter = 100000
            for i, l in enumerate(range(0, len(texts), itter)):
                process = []

                docs = texts[l:l+itter]
                t_len = len(docs)
                t_itter = t_len//self.n_process

                for k, j in enumerate(range(0, t_len, t_itter)):
                    process.append(tokenizer_process_init(i*self.n_process+k, docs[j:j+t_itter]))

                print("[MULTIPROCESS LOOP] Starting", self.n_process, "process")
                for p in process:
                    p.start()

                print("[MULTIPROCESS LOOP] Wait", self.n_process, "process")
                for p in process:
                    p.join()

                del docs
                gc.collect()

            # merge the tokenizer
            print("[TOKENIZER] Merge tokenized files")
            files = sorted(os.listdir(merge_tokenizer_path))
            del texts

            for file in files:
                log.info("[TOKENIZER] Load {}".format(file))
                with open(os.path.join(merge_tokenizer_path, file), "rb") as f:
                    tokenized_texts.extend(pickle.load(f))

        except Exception as e:
            raise e
        finally:
            # always remove the temp directory
            log.info("[TOKENIZER] Remove {}".format(merge_tokenizer_path))
            shutil.rmtree(merge_tokenizer_path)

        return tokenized_texts
