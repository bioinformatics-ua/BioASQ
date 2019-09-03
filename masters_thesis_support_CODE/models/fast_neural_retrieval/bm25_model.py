import pickle
import os
import gc
import heapq

from multiprocessing import Process, Pool

from models.generic_model import ModelAPI
from pubmed_data import pubmed_helper as ph


from models.fast_neural_retrieval.bm25_inverted_index import InvertedIndex, DocumentLengthTable
from models.fast_neural_retrieval.bm25_score import score_BM25

import time

class BM25(ModelAPI):

    def __init__(self, num_threads=4, num_index_files=5, full_tokens = False, saved_models_path=None):
        if saved_models_path is None:
            super().__init__()
        else:
            super().__init__(saved_models_path=saved_models_path)

        self.document_len_table = DocumentLengthTable()
        self.num_threads = num_threads
        self.num_index_files = num_index_files
        self.full_tokens = full_tokens
        self.tokenizer_mode = "bllip_stem" + ("_full_tokens" if self.full_tokens else "")

    def _training_process(self, data):
        
        print("Note: The training process for the BM25, will consist in the build of the inverted index and doc table")
        
        def __multi_process(strat_doc_id,documents):
            print("Thread strat",os.getpid())
            inverted_index = InvertedIndex()
            document_len_table = DocumentLengthTable() 
        
            for i,document in enumerate(documents):
                #build inverted index
                for token in document:

                    inverted_index.add(token, strat_doc_id+i)

                document_len_table.add(len(document), strat_doc_id+i)
                
            file_name_inverted_index = "/backup/saved_models/bm25_to_merge/inverted_index_{0:08}".format(strat_doc_id)
            file_name_document_len = "/backup/saved_models/bm25_to_merge/document_len_{0:08}".format(strat_doc_id)
            
            print("save:",file_name_inverted_index)
            with open(file_name_inverted_index,"wb") as file:
                #json.dump(inverted_index,file)
                pickle.dump(inverted_index,file)
            
            print("save:",file_name_document_len)
            with open(file_name_document_len,"wb") as file: 
                pickle.dump(document_len_table,file)
            
            print("Thread End",os.getpid())
        
        document_start_index = 0
        document_cumulative_index = 0
        # If the input is a generator
        if isinstance(data, types.GeneratorType):
            for documents in data:
                docs_per_thread = len(documents)//self.num_threads
                batch = list(range(0,len(documents),docs_per_thread))
                
                if len(batch)==self.num_threads:
                    batch.append(len(documents))
                else:
                    batch[self.num_threads] = len(documents)
                
                #Run multithread
                threads = []
                
                for i in range(self.num_threads):
                    #documents[batch[i]:batch[i+1]]
                    
                    threads.append(Process(target=__multi_process, args=(document_start_index, documents[batch[i]:batch[i+1]],)))
                    document_start_index = document_cumulative_index + batch[i+1]
                
                document_cumulative_index = document_start_index
                
                for t in threads:
                    t.start()
                
                print("Wait for the threads")
                for t in threads:
                    t.join()
            
            print("Document table merge")
            path_root = "/backup/saved_models/bm25_to_merge/"
            
            for f_name in sorted(filter(lambda x:"document" in x,os.listdir(path_root))):
                #verify
                print("open file",f_name)
                with open(os.path.join(path_root,f_name),"rb") as f:
                    for docid, token_len in pickle.load(f).table.items():
                        self.document_len_table.table[docid] = token_len
            
            print("Start the inverted_index merge")
            #Load tokenizer to get word_counts
            #merge inverted index
            
            tk = ph.load_tokenizer(mode="bllip_stem")
            if self.full_tokens:
                tk.num_words = None
                valid_tokens = map(lambda x:(tk.word_index[x[0]],x[1]),tk.word_counts.items())
            else:
                #for this tokenizer only the top 20 were used
                valid_tokens = map(lambda x:(tk.word_index[x[0]],x[1]),filter(lambda x:x[1]>=20,tk.word_counts.items()))
            
            print("Sorting token_word_count")
            #remove the first token that is the "."
            token_freq = list(sorted(valid_tokens, key = lambda x:-x[1]))[1:]
            print("End Sort")
            total_tokens = sum(map(lambda x:x[1],token_freq))
            
            #remove tokenizer
            del tk
            
            num_tokens_per_index = total_tokens//self.num_index_files

            file_division = [[] for _ in range(self.num_index_files)]

            count = 0
            index_current_token = 0
            for i in range(self.num_index_files):

                while count<num_tokens_per_index and index_current_token<len(token_freq):

                    file_division[i].append(token_freq[index_current_token][0])
                    count += token_freq[index_current_token][1]
                    index_current_token += 1

                count=0
            
            print("Invert index division:",list(map(lambda x:len(x),file_division)))
            
            files_to_merge = sorted(filter(lambda x:"index" in x,os.listdir(path_root)))

            for i in range(self.num_index_files):
                print("BUILD: index file",i)
                inverted_index = {}

                for f_name in files_to_merge:

                    print("open file",f_name)
                    with open(os.path.join(path_root,f_name),"rb") as f:
                        loadded_inv_index = pickle.load(f).index

                    for token in file_division[i]:
                        if token in loadded_inv_index:
                            if token not in inverted_index:
                                inverted_index[token] = loadded_inv_index[token]
                            else:
                                inverted_index[token].update(loadded_inv_index[token])
                    
                    del loadded_inv_index
                
                file_name = "inverted_index_"+str(i)+".p"
                print("Saving:",file_name)
                with open(os.path.join(self.saved_models_path, "bm25_data", file_name),"wb") as f:
                    pickle.dump(inverted_index,f)
        
            
        else:
            raise RuntimeError("Only work with data generators")

            
    
    
    def _predict_process(self, queries, **kwargs):
        """
        queries - List of queries, here each is represented by json {id:<id>, body:<string>}
        
        """
        if 'use_precomputed_score' in kwargs:
            use_precomputed_score = kwargs.pop('use_precomputed_score')
        else:
            use_precomputed_score = False
        
        if 'mapping_f' in kwargs:
            mapping_f = kwargs.pop('mapping_f')
        else:
            mapping_f = None
        
        if 'top_k' in kwargs:
            top_k = kwargs.pop('top_k')
        else:
            top_k = 10000
        
        if kwargs:
            raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))

        #load tokenizer
        tk = ph.load_tokenizer(mode = self.tokenizer_mode)

        if self.full_tokens:
            tk.num_words = None
        
        print("Precomputed score:","Enable" if use_precomputed_score else "Disable")
        print("Using map function:","Disable" if mapping_f is None else "Enable")
        
        results = { query["id"]:{"body":query["body"],"documents":{}} for query in queries }
        
        doc_N = len(self.document_len_table)
        avgdl = self.document_len_table.get_average_length()    
            
        for i in range(self.num_index_files):
            if use_precomputed_score:
                index_file_name = "inverted_index_bm25_"+str(i)+".p"
            else:
                index_file_name = "inverted_index_"+str(i)+".p"
            print("Load",index_file_name)
            
            bm25_start_load_index = time.time()
            
            with open(os.path.join(self.saved_models_path,"bm25_data",index_file_name),"rb") as f:
                inverted_index = InvertedIndex(index=pickle.load(f))
            
            print("Time load index",time.time()-bm25_start_load_index)
            
            for j,query in enumerate(queries):
                if j%10==0:
                    print("query",j,end="\r")
                self.__single_query(tk.texts_to_sequences([query["body"]])[0], 
                                    results[query["id"]]["documents"], 
                                    inverted_index,
                                    doc_N, 
                                    avgdl, 
                                    use_precomputed_score)

                
            print("Number of matching documents to query", query["id"], len(results[query["id"]]["documents"]))

            del inverted_index
            gc.collect() #force garbage collector
            
        print("Sorting the results")
        #TODO CHANGE TO: heapq.nlargest(k, dictionary, key=dictionary.get)
        
        bm25_start_sort = time.time()
        for k,v in results.items():
            results[k]["documents"] = heapq.nlargest(top_k, v["documents"].items(), key=lambda x:x[1])
        print("Time sort results",time.time()-bm25_start_sort)
        #del results
        gc.collect() #force garbage collector
                                             
        return results
        
    
    def __single_query(self, query, query_result, inverted_index, doc_N, avgdl, use_precomputed_score):
        
        #if the query_result is empty, the doc_dict can be copied to the results
        if len(query_result)==0:
            
            #get low index term of the query to maximize efficience
            min_index_query_term = min(query)
            
            #this term is in the index so copy the doc_dict to the results
            if min_index_query_term in inverted_index:
                query_result.update(inverted_index[min_index_query_term])
                query.remove(min_index_query_term)

        
        for term in query:
            #print(term,":",term in inverted_index)
            #search
            if term in inverted_index:
                doc_dict = inverted_index[term] # retrieve index entry
                
                for docid, freq in doc_dict.items(): #for each document and its word frequency
                    
                    if use_precomputed_score:
                        score = freq
                    else:
                        score = score_BM25(n=len(doc_dict), 
                                           f=freq, 
                                           qf=1, 
                                           r=0, 
                                           N=doc_N,
                                           dl=self.document_len_table.get_length(docid), 
                                           avdl=avgdl) # calculate score
                    
                    if docid in query_result: #this document has already been scored once
                        query_result[docid] += score
                    else:
                        query_result[docid] = score
        
        #print(len(query_result))
                        
    def pre_compute_index_bm25(self, qf=1,r=0):
        
        doc_N = len(self.document_len_table)
        avgdl = self.document_len_table.get_average_length()
        
        
        for i in range(self.num_index_files):
            index_file_name = "inverted_index_"+str(i)+".p"
            print("Load",index_file_name)
            with open(os.path.join(self.saved_models_path,"bm25_data",index_file_name),"rb") as f:
                inverted_index = InvertedIndex(index=pickle.load(f))
            
            for term, doc_dict in inverted_index.index.items():
                
                for docid, freq in doc_dict.items(): 
                    
                    inverted_index[term][docid] = score_BM25(n=len(doc_dict), 
                                                               f=freq, 
                                                               qf=qf, 
                                                               r=r, 
                                                               N=doc_N,
                                                               dl=self.document_len_table.get_length(docid), 
                                                               avdl=avgdl) # calculate score
            
            index_file_name = "inverted_index_bm25_"+str(i)+".p"
            print("Save",index_file_name)
            
            with open(os.path.join(self.saved_models_path,"bm25_data",index_file_name),"wb") as f:
                pickle.dump(inverted_index.index,f)
                
            

            #results[j] = dict(list(map(lambda x:(self.mapping(x[0]),x[1]),results[j].items())))
            
            del inverted_index
        
    @staticmethod
    def load(path = '/backup/saved_models/',full_tokens = False):
        table = {}
        try:
            with open(os.path.join(path, "bm25_data", "document_len.p"),"rb") as file:
                table = pickle.load(file)
        except e:
            print(e)
            print("Error when loading the model, a non trained model will be returned")
        
        bm25 = BM25(full_tokens=full_tokens)
        bm25.saved_models_path = path
        
        if len(table)!=0:
            bm25.document_len_table.table = table
            bm25.trained = True
        
        return bm25
    
    def save(self):
        with open(os.path.join(self.saved_models_path, "bm25_data", "document_len.p"),"wb") as file:
            pickle.dump(self.document_len_table.table,file)
        
