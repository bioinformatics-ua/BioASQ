import os
import pickle
import numpy as np

from models.generic_model import ModelAPI
from pubmed_data import pubmed_helper as ph
from models.deep_model_for_ir.custom_layers import SimilarityMatrix, MaskedConv2D, TermGatingDRMM_FFN, SelfAttention

from tensorflow import unstack, stack
##Test 
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers, activations
from tensorflow.keras.initializers import Zeros, Ones
from tensorflow.keras.layers import Dense, Lambda, Dot,Reshape, Concatenate, Layer, Embedding, Input, Conv2D, GlobalMaxPooling2D, Flatten, TimeDistributed, GRU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.activations import tanh, sigmoid

from tensorflow.keras.preprocessing.sequence import pad_sequences

#Number max of term per query
MAX_Q_TERM = 13

#Number max of the snippet terms
QUERY_CENTRIC_CONTEX = 15

#Number max of passages per query term
MAX_PASSAGES_PER_QUERY = 5

#Snippet position padding value
SNIPPET_POSITION_PADDING_VALUE = -1


class DeepRankV4(ModelAPI):
    
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

        #Mode for the creation of the S matrix
        S_MATRIX_MODE = 0
        #S_MATRIX_DIMENSION = EMB_DIM*2+1

        #Train embedding weights
        EMB_TRAINABLE = False

        #Number of filters in CNN
        CNN_FILTERS = 100
        CNN_KERNELS = (3,3)

        #RNN DIM
        USE_BIDIRECTIONAL = False
        GRU_REPRESENTATION_DIM = 58

        ACTIVATION_FUNCTION = "selu"

        REGULARIZATION = regularizers.l2(0.0001)

        #Term gating network mode
        TERM_GATING_MODE =  3#2- weigt fixed per position, 1 - DRMM like term gating

        assert S_MATRIX_MODE in [0,1]
        assert TERM_GATING_MODE in [0,1,2,3]

        #MACRO STYLE
        S_MATRIX_3D_DIMENSION = 1 if S_MATRIX_MODE==0 else self.emb_size*2+1

        print("\n\n\tInput Layer Model\n\n")
        
        
        """""""""""""""""""""""""""
           ---- Input Layer ----
        """""""""""""""""""""""""""
        #Embedding Layer
        embedding = Embedding(self.vocab_size, self.emb_size, name="embedding_layer", trainable=EMB_TRAINABLE)

        #S matrix ref in the paper
        similarity_matrix = SimilarityMatrix(MAX_Q_TERM, 
                                             QUERY_CENTRIC_CONTEX, 
                                             interaction_mode=S_MATRIX_MODE, 
                                             name="query_snippet_similarity")

        #transpose (None, QUERY_CENTRIC_CONTEX, EMB_DIM) => (None, EMB_DIM, QUERY_CENTRIC_CONTEX) 
        transpose_layer = Lambda(lambda x:K.permute_dimensions(x,[0,1,2,4,3]), name="snippet_transpose") 

        #Snippet single embedding transformation
        snippet_token_input = Input(shape = (MAX_Q_TERM, MAX_PASSAGES_PER_QUERY, QUERY_CENTRIC_CONTEX,), name = "snippet_token")
        snippet_emb = embedding(snippet_token_input)
        snippet_emb_transpose = transpose_layer(snippet_emb)
        snippet_emb_model = Model(inputs = [snippet_token_input], outputs=[snippet_emb_transpose], name = "snippet_emb_model")
        print("\n\nsnippet_emb_model summary")
        snippet_emb_model.summary()
        
        
        print("\n\n\tMeasure Model\n\n")
        """""""""""""""""""""""""""
           ---- Measure Layer ----
        """""""""""""""""""""""""""
        
        #Exctrate high-level features from query and snippet interactions with CNN
        cnn_extraction_model = Sequential(name="cnn_extraction_model")
        cnn_extraction_model.add(MaskedConv2D(input_shape = (MAX_Q_TERM, QUERY_CENTRIC_CONTEX, S_MATRIX_3D_DIMENSION), filters = CNN_FILTERS, kernel_size=CNN_KERNELS, activation=ACTIVATION_FUNCTION ))
        cnn_extraction_model.add(GlobalMaxPooling2D())
        print("\n\ncnn_extraction_model summary")
        cnn_extraction_model.summary()


        td_cnn_extraction_model = Sequential(name="TD_cnn_extraction_model")
        td_cnn_extraction_model.add(TimeDistributed(cnn_extraction_model, input_shape=(MAX_PASSAGES_PER_QUERY, MAX_Q_TERM, QUERY_CENTRIC_CONTEX, S_MATRIX_3D_DIMENSION)))
        td_cnn_extraction_model.summary()

        """""""""""""""""""""""""""
             ---- Layers ----
        """""""""""""""""""""""""""
        #concatenation layer over the last dimension
        concat_snippet_position = Concatenate( name = "concat_snippet_position")

        #Attention
        self_attention = SelfAttention(CNN_FILTERS)
    

        #add dimension Layer
        add_passage_dim = Lambda(lambda x:K.expand_dims(x,axis=1), name="add_passage_dim")#Reshape(target_shape=(1,GRU_REPRESENTATION_DIM))

        #add last dimension Layer
        add_dim = Lambda(lambda x:K.expand_dims(x), name="add_dim")

        #reciprocal function
        reciprocal_f = Lambda(lambda x:1/(x+2), name="reciprocal_function")

        #concatenation layer over second dimension (passage dimension)
        concat_representation = Concatenate(axis = 1,name = "concat_representation")

        
        print("\n\n\t Aggregation Model\n\n")
        """""""""""""""""""""""""""
           - Aggregation Layer -
        """""""""""""""""""""""""""

        query_token_input = Input(shape=(MAX_Q_TERM,), name="ds_query_tokens")
        doc_score_snippet_input = Input(shape = (MAX_Q_TERM,MAX_PASSAGES_PER_QUERY,QUERY_CENTRIC_CONTEX), name = "ds_snippet_tokens")
        doc_score_snippet_position_input = Input(shape = (MAX_Q_TERM,MAX_PASSAGES_PER_QUERY), name = "ds_snippet_position_tokens")


        unstack_by_q_term = Lambda(lambda x:unstack(x,axis=1), name="unstack_query_term")

        query_emb = embedding(query_token_input)

        doc_score_snippet_emb = embedding(doc_score_snippet_input)
        doc_score_snippet_emb_transpose = transpose_layer(doc_score_snippet_emb)

        query_snippets_s_matrix = similarity_matrix([query_emb,doc_score_snippet_emb_transpose])

        list_of_s_matrix_by_q_term = unstack_by_q_term(query_snippets_s_matrix)
        list_of_snippet_postion_by_q_term = unstack_by_q_term(doc_score_snippet_position_input)

        relevance_representation = []
        for i in range(MAX_Q_TERM):

            snippet_relative_position = reciprocal_f(list_of_snippet_postion_by_q_term[i])

            local_relevance = td_cnn_extraction_model(list_of_s_matrix_by_q_term[i])

            local_relevance_position = concat_snippet_position([local_relevance,add_dim(snippet_relative_position)])

            relevance_representation.append(add_passage_dim(self_attention(local_relevance_position)))

        concat_relevance = concat_representation(relevance_representation)

        snippet_rnn_rep_dim = CNN_FILTERS
        
        term_gating = TermGatingDRMM_FFN(embedding_dim = self.emb_size, 
                                         rnn_dim = snippet_rnn_rep_dim,
                                         activation=ACTIVATION_FUNCTION, 
                                         regularizer=REGULARIZATION)
        
        document_score = term_gating([query_emb,concat_relevance])

        self.document_score_model = Model(inputs = [query_token_input, doc_score_snippet_input, doc_score_snippet_position_input], outputs = [document_score], name="query_document_score")
        print("\n\n Document Model \n\n")
        self.document_score_model.summary()
        
        #was saved in train format
        
    def _training_process(self, data):
        print("Use NOTEBOOK to train this model instead")
        self._training_process()
    
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

        for query_id, query_data in queries.items():
            query = []
            query_doc = []
            query_doc_position = []
            tokenized_query = self.tokenizer.texts_to_sequences([query_data["body"]])[0]
            #manualy remove the stopwords
            tokenized_query = [ token for token in tokenized_query if token not in self.biomedical_stop_words_tokens]

            tokenized_query = pad_sequences([tokenized_query], maxlen = MAX_Q_TERM, padding="post")[0]

            for doc_index in query_data["documents"]:
                #positive

                tokenized_doc = self.tokenizer.texts_to_sequences([collection[doc_index[0]]])[0]
                doc_snippets, doc_snippets_position = self.__snippet_interaction(tokenized_query, tokenized_doc)
                ### add ###

                query.append(tokenized_query)

                #positive doc
                query_doc.append(doc_snippets)
                query_doc_position.append(doc_snippets_position)


            #missing fill the gap for the missing query_terms

            X = [np.array(query), np.array(query_doc), np.array(query_doc_position)]
            
            print("Predict query:",len(queries_deep_rank_result),end="\r")
            
            deep_ranking = self.document_score_model.predict(X)
            
            deep_ranking = map(lambda x:x[0],deep_ranking.tolist())
            bm25_results = query_data["documents"]
            deep_ranking_pmid = list(zip(bm25_results,deep_ranking))
            
            deep_ranking_pmid.sort(key=lambda x:-x[1])
            
            deep_ranking_pmid = list(map(lambda x:map_f(x[0][0]), deep_ranking_pmid))
            
            queries_deep_rank_result.append({"id":query_id,"documents":deep_ranking_pmid})
            
        return queries_deep_rank_result    
    
    def __snippet_interaction(self, tokenized_query, tokenized_doc, snippet_length=QUERY_CENTRIC_CONTEX):
        
        snippets = []
        snippets_position = [] 

        half_size = snippet_length//2
        
        #O(n^2) complexity, probably can do better with better data struct TODO see if is worthit
        for query_token in tokenized_query:
            
            snippets_per_token = []
            snippets_per_token_position = []
            
            if query_token != 0: #jump padded token
            
                for i,doc_token in enumerate(tokenized_doc):

                    if doc_token==query_token:

                        lower_index = i-half_size
                        lower_index = max(0,lower_index)

                        higher_index = i+half_size
                        higher_index = min(len(tokenized_doc),higher_index)

                        snippets_per_token.append(tokenized_doc[lower_index:higher_index])
                        snippets_per_token_position.append(i)
            
            if len(snippets_per_token)==0:
                snippets.append(np.zeros((MAX_PASSAGES_PER_QUERY,QUERY_CENTRIC_CONTEX), dtype=np.int32))
                snippets_position.append(np.zeros((MAX_PASSAGES_PER_QUERY), dtype=np.int32)+SNIPPET_POSITION_PADDING_VALUE)
                continue
                
            max_snippets_len = min(MAX_PASSAGES_PER_QUERY, len(snippets_per_token))
            
            ### snippets in matrix format
            #pad
            snippets_per_token = pad_sequences(snippets_per_token, maxlen = QUERY_CENTRIC_CONTEX, padding="post")
            #fill the gaps
            _temp = np.zeros((MAX_PASSAGES_PER_QUERY,QUERY_CENTRIC_CONTEX), dtype=np.int32)
            _temp[:max_snippets_len] = snippets_per_token[:max_snippets_len]
            snippets.append(_temp)
            
            ### snippets_position in matrix format
            #pad
            snippets_per_token_position = pad_sequences([snippets_per_token_position], maxlen = MAX_PASSAGES_PER_QUERY, padding="post", value=SNIPPET_POSITION_PADDING_VALUE)[0]
            snippets_position.append(snippets_per_token_position)
            
        return snippets, snippets_position
    @staticmethod
    def load(path = '/backup/saved_models/deep_rankV5'):
        
        with open(os.path.join(path, "meta_data.p"),"rb") as file:
            meta_data = pickle.load(file)
        
        deep_rank_model = DeepRankV4(meta_data["vocab_size"],
                                     meta_data["emb_size"],
                                     meta_data["tk_mode"],
                                     path)
        
        #load weights
        deep_rank_model.document_score_model.load_weights('/backup/saved_models/deep_rankV5/deep_rank_weights.h5')

        deep_rank_model.trained = True
       
        return deep_rank_model
