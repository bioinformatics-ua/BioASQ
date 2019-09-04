import os
import pickle
import numpy as np

import time

from models.generic_model import ModelAPI
from pubmed_data import pubmed_helper as ph
from models.deep_model_for_ir.custom_layers import SelfAttention, CrossAttention

from tensorflow import unstack, stack
##Test 
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers, activations
from tensorflow.keras.initializers import Zeros, Ones
from tensorflow.keras.layers import Dense, Lambda, Bidirectional, Dot,Masking,Reshape, Concatenate, Layer, Embedding, Input, Conv2D, GlobalMaxPooling2D, Flatten, TimeDistributed, GRU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.activations import tanh, sigmoid


from tensorflow.keras.preprocessing.sequence import pad_sequences

#Number max of term per query
MAX_Q_TERM = 13

#Number max of the snippet terms
MAX_SNIPPET_LENGTH = 13

MAX_NUMBER_SNIPPETS = 20

MAX_DOCUMENT_TOKENS = MAX_SNIPPET_LENGTH*MAX_NUMBER_SNIPPETS

ATTENTION_DIMENSION = 100

assert ATTENTION_DIMENSION%2==0

GRU_DIM = ATTENTION_DIMENSION//2

NUM_OF_SELF_ATTENTION = 1

#Train embedding weights
EMB_TRAINABLE = False

ACTIVATION_FUNCTION = "selu"


class HAR(ModelAPI):
    
    def __init__(self, vocab_size, emb_size, tokenizer_mode="regex", saved_models_path=None):
        if saved_models_path is None:
            super().__init__()
        else:
            super().__init__(saved_models_path=saved_models_path)
        
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.tokenizer_mode = tokenizer_mode
        self.tokenizer = ph.load_tokenizer(mode = tokenizer_mode)
        #hardcode stop words
        biomedical_stop_words = ["a", "about", "again", "all", "almost", "also", "although", "always", "among", "an", "and", "another", "any", "are", "as", "at", "be", "because", "been", "before", "being", "between", "both", "but", "by", "can", "could", "did", "do", "does", "done", "due", "during", "each", "either", "enough", "especially", "etc", "for", "found", "from", "further", "had", "has", "have", "having", "here", "how", "however", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "kg", "km", "made", "mainly", "make", "may", "mg", "might", "ml", "mm", "most", "mostly", "must", "nearly", "neither", "no", "nor", "obtained", "of", "often", "on", "our", "overall", "perhaps", "pmid", "quite", "rather", "really", "regarding", "seem", "seen", "several", "should", "show", "showed", "shown", "shows", "significantly", "since", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "then", "there", "therefore", "these", "they", "this", "those", "through", "thus", "to", "upon", "use", "used", "using", "various", "very", "was", "we", "were", "what", "when", "which", "while", "with", "within", "without", "would"]
        self.biomedical_stop_words_tokens = set(self.tokenizer.texts_to_sequences([biomedical_stop_words])[0])
        
        self._build_model()
        
    def _build_model(self):
        print("Build Model")
        
        K.clear_session()

        input_query = Input(shape=(MAX_Q_TERM,), name = "query_input")
        input_doc = Input(shape=(MAX_NUMBER_SNIPPETS, MAX_SNIPPET_LENGTH), name = "document_input")
        #input_doc = [Input(shape=(MAX_SNIPPET_LENGTH,), name = "document_input_"+j) for j in range(MAX_NUMBER_SNIPPETS)]

        unstack_snippets = Lambda(lambda x:unstack(x,axis=1), name="unstack_snippets")

        input_docs = unstack_snippets(input_doc)

        emb_layer = Embedding(self.vocab_size, self.emb_size,name="embedding_layer", trainable=EMB_TRAINABLE)

        context_encoder = Bidirectional(GRU(GRU_DIM, return_sequences=True))

        query_self_attention = SelfAttention(ATTENTION_DIMENSION)
        cross_layer = CrossAttention()
        snippet_lvl_1_self_attention = SelfAttention(ATTENTION_DIMENSION)
        snippet_lvl_2_self_attention = SelfAttention(ATTENTION_DIMENSION)

        snippet_expand_dim = Lambda(lambda x:K.expand_dims(x,axis=1))
        snippet_concat = Lambda(lambda x:K.concatenate(x,axis=1))

        snippets_projection = Dense(ATTENTION_DIMENSION)

        q_doc_mult = Lambda(lambda x:x[0]*x[1])

        #NO ACTIVATION??
        fnn_h1 = Dense(ATTENTION_DIMENSION)
        fnn_h2 = Dense(ATTENTION_DIMENSION//2)
        fnn_h3 = Dense(1)

        """
        ASSEMBLE
        """

        query_emb = emb_layer(input_query)
        query_context_aware = context_encoder(query_emb)
        query_weighted = query_self_attention(query_context_aware)

        snippets = []
        for input_snippet in input_docs:
            snippet_emb = emb_layer(input_snippet)
            snippet_context_aware = context_encoder(snippet_emb)
            snippet_query = cross_layer([query_context_aware, snippet_context_aware])
            snippet_weighted = snippet_lvl_1_self_attention(snippet_query)
            snippets.append(snippet_expand_dim(snippet_weighted))

        snippets = snippet_concat(snippets)
        snippets = snippet_lvl_2_self_attention(snippets)
        snippets = snippets_projection(snippets)
        q_snippets_rep = q_doc_mult([query_weighted, snippets])
        score = fnn_h1(q_snippets_rep)
        score = fnn_h2(score)
        score = fnn_h3(score)

        inputs = [input_query, input_doc]

        self.document_score_model = Model(inputs=inputs, outputs = [score])

        self.document_score_model.summary()
        
        #was saved in train format
        
    def _training_process(self, data):
        print("Use NOTEBOOK to train this model instead")
        super()._training_process()
    
    def _prepare_data(self, query_body, documents, collection):
        
        query = []
        query_doc = []
        
        tokenized_query = self.tokenizer.texts_to_sequences([query_body])[0]
        #manualy remove the stopwords
        tokenized_query = [ token for token in tokenized_query if token not in self.biomedical_stop_words_tokens]

        tokenized_query = pad_sequences([tokenized_query], maxlen = MAX_Q_TERM, padding="post")[0]

        for doc_index in documents:
            #positive

            tokenized_doc = self.tokenizer.texts_to_sequences([collection[doc_index[0]]])[0]
            doc_sentences = self.__snippet_split(tokenized_doc)
            ### add ###

            query.append(tokenized_query)

            #positive doc
            query_doc.append(doc_sentences)


        #missing fill the gap for the missing query_terms

        return [np.array(query), np.array(query_doc)]
    
    def _predict_process(self, queries, **kwargs):
        """
        queries: {"id":{"body":"<string>","documents":[index]}}
        """
        print("Start DR predict")
        if 'collection' in kwargs:
            collection = kwargs.pop('collection')
        else:
            raise TypeError('missing collection param')
        
        if 'map_f' in kwargs:
            map_f = kwargs.pop('map_f')
        else:
            raise TypeError('missing collection map_f')
        
        if kwargs:
            raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))
        
        
        queries_deep_rank_result = []
        predict_times = []
        sort_times = []
        
        
        for query_id, query_data in queries.items():            

            X = self._prepare_data(query_data["body"], query_data["documents"], collection)
            
            print("Predict query:",len(queries_deep_rank_result),end="\r")
            
            start_q_time = time.time()
            deep_ranking = self.document_score_model.predict(X)
            predict_times.append(time.time()-start_q_time)
            
            deep_ranking = map(lambda x:x[0],deep_ranking.tolist())
            bm25_results = query_data["documents"]
            deep_ranking_pmid = list(zip(bm25_results,deep_ranking))
            
            start_sort_time = time.time()
            deep_ranking_pmid.sort(key=lambda x:-x[1])
            sort_times.append(time.time()-start_sort_time)
            
            deep_ranking_pmid = list(map(lambda x:"http://www.ncbi.nlm.nih.gov/pubmed/"+map_f(x[0][0]), deep_ranking_pmid))
            
            queries_deep_rank_result.append({"id":query_id,"documents":deep_ranking_pmid})
            
        print("Prediction avg time",np.mean(predict_times), "Sort avg time",np.mean(sort_times))
        
        return queries_deep_rank_result    
    
    def __snippet_split(self, tokenized_doc, snippet_length=MAX_SNIPPET_LENGTH):
        

        if len(tokenized_doc) < MAX_DOCUMENT_TOKENS:
            #pad
            tokenized_doc += [0]*(MAX_DOCUMENT_TOKENS-len(tokenized_doc))
            
        else:
            tokenized_doc = tokenized_doc[:MAX_DOCUMENT_TOKENS] #cut
        
        index_list = list(range(0, MAX_DOCUMENT_TOKENS, MAX_SNIPPET_LENGTH))+[MAX_DOCUMENT_TOKENS]
        #print(tokenized_doc)
        snippets = [ tokenized_doc[index_list[i]:index_list[i+1]] for i in range(len(index_list)-1)  ]
        #print(snippets)
        return snippets
    
    @staticmethod
    def load(path = '/backup/saved_models/har_V1'):
        
        with open(os.path.join(path, "meta_data.p"),"rb") as file:
            meta_data = pickle.load(file)
        
        har = HAR(meta_data["vocab_size"],
                     meta_data["emb_size"],
                     meta_data["tk_mode"],
                     path)
        
        #load weights
        har.document_score_model.load_weights('/backup/saved_models/har_V1/har_model_weights.h5')

        har.trained = True
       
        return har
