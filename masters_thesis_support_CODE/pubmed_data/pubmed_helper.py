import os
import tarfile
import codecs
import json
import gc
import pickle
import sys
import sentencepiece as spm

#ensure back-compatibility
if os.getcwd().split("/")[-1] == "bioASQ-taskb":
    print("import")
    module_path = os.path.abspath(os.path.join('.','pubmed_data'))
    if module_path not in sys.path:
        sys.path.append(module_path)


def read_pubmed_collection(path="/backup/pubmed_archive_json/pubmed_dump.tar.gz"):
    """
    Read tar.gz file with pubmed dump
    """

    reader = codecs.getreader("utf-8")
    articles = []
    tar = tarfile.open(path)
    print("Preparing the tar.gz... should take few minutes")
    for m in tar.getmembers():
        print("File",m.name,":",end="")
        f = tar.extractfile(m)
        articles.extend(json.load(reader(f)))
        f.close()
        
        del f
        print(" Done")
        
    del reader
    del tar
    
    return articles

def create_pubmed_collection_generator(mapping_function = None,path="/backup/pubmed_archive_json/pubmed_ready.tar.gz"):
    """
    return a generator function
    """
    
    reader = codecs.getreader("ascii")
    tar = tarfile.open(path)
    
    print("Open",path)
    members = tar.getmembers()
    print("Creating generator")
    
    def generator():
        for m in members:
            print("Open the file:",m.name)
            f = tar.extractfile(m)
            #Force the mapping
            articles = json.load(reader(f))
            if mapping_function is not None:
                articles = list(map(mapping_function, articles))

            print("Returning:",len(articles),"articles")
            yield articles
            
            f.close()
            del f
            
            print("Force garbage collector",gc.collect())
            
        #Clean up the tar file?
        #print("Clean up!")
        #del tar
        #gc.collect()
            
    return generator
    
def create_tokenized_pubmed_collection_generator(path=None,mode="keras"):
    
    if not path:
  
        if mode == "bllip_stem_N20":
            path = "/backup/pubmed_archive_tokenized/bllip_stem_N20_title_abs.tar.gz"
        elif mode == "bllip_stem_full_tokens":
            path = "/backup/pubmed_archive_tokenized/bllip_stem_full_tokens_title_abs.tar.gz"
        elif mode == "regex_full_tokens":
            path = "/backup/pubmed_archive_tokenized/regex_full_tokens_title_abs.tar.gz"
        elif mode == "regex_less_700k_freq":
            path = "/backup/pubmed_archive_tokenized/regex_less_700k_freq_title_abs.tar.gz"
        else:
            raise TypeError('path or mode unvailable')
    
    
    
    tar = tarfile.open(path)
    
    print("Open",path)
    members = tar.getmembers()
    print("Creating generator")
    
    def generator():
        for m in members:
            print("Open the file:",m.name)
            f = tar.extractfile(m)
            tokenized_articles = pickle.load(f)
            print("Returning:",len(tokenized_articles),"articles")
            
            yield tokenized_articles
            f.close()
            del f
            del tokenized_articles
            
            print("Force garbage collector",gc.collect())
    
    
    return generator

def load_tokenizer(mode):
    
    if mode=="bllip_stem_full_tokens":
        path = "bllip_stem_full_tokens_tokenizer.p"
    elif mode=="bllip_stem_N20":
        path = "bllip_stem_N20_tokenizer.p"
    elif mode=="hashtrick_full_tokens":
        path = "hashtrick_full_tokens_tokenizer.p"
    elif mode=="regex_full_tokens":
        path = "regex_full_tokens_tokenizer.p"
    elif mode=="regex_less_700k_freq":
        path = "regex_less_700k_freq_tokenizer.p"
    else:
        raise TypeError('path or mode unvailable')
    
    print("Load",path)
    with open(os.path.join("/backup/pubmed_tokenizers/",path),"rb") as f:
        return pickle.load(f)

    
def load_embeddings(mode):
    
    if mode=="regex_full_tokens":
        path = "regex_full_tokens_word_embedding.p"
    elif mode == "regex_less_700k_freq":
        path = "regex_less_700k_freq_embedding.p"
    else:
        raise TypeError('mode unvailable')
        
    print("Load",path)
    with open(os.path.join("/backup/pre-trained_embeddings/word_emb_by_fasttext/",path),"rb") as f:
        return pickle.load(f)

def pmid_index_mapping():
    path = "/backup/saved_models/pmid_index_mapping.p"
    print("Load",path)
    with open(path,"rb") as f:
        return pickle.load(f)


def load_bpe_model(model_name):
    model_name = os.path.join("/backup/bpe_model",model_name+".model")
    sp_bpe = spm.SentencePieceProcessor()
    sp_bpe.load(model_name)
    return sp_bpe