from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Conv2D, Dense
from tensorflow.keras import initializers, regularizers, activations

class SimilarityMatrix(Layer):   
    def __init__(self, query_max_term, snippet_max_term, interaction_mode=0, **kwargs):
        """
        interaction mode 0: only use similarity matrix
                    mode 1: similarity matrix + query and snippet embeddings
        """
        assert interaction_mode in [0,1] #only valid modes

        self.query_max_term = query_max_term
        self.snippet_max_term = snippet_max_term
        self.interaction_mode = interaction_mode

        super().__init__(**kwargs)

    def call(self,x):
        if self.interaction_mode==0:
            #sim => dot product (None, MAX_Q_TERM, EMB_DIM) x (None, MAX_Q_TERM, MAX_PASSAGE_PER_Q, EMB_DIM, QUERY_CENTRIC_CONTEX)
            query = K.expand_dims(x[0], axis=1) #(None, 1, MAX_Q_TERM, EMB_DIM)
            query = K.expand_dims(query, axis=1) #(None, 1, 1, MAX_Q_TERM, EMB_DIM)
            query = K.repeat_elements(query,x[1].shape[1],axis=1) #(None, MAX_PASSAGE_PER_Q, MAX_Q_TERM, EMB_DIM)
            query = K.repeat_elements(query,x[1].shape[2],axis=2)
            s_matrix = K.batch_dot(query,x[1]) #(None, MAX_PASSAGE_PER_Q, MAX_Q_TERM, #(None, MAX_PASSAGE_PER_Q, MAX_Q_TERM, EMB_DIM)

            s_matrix = K.expand_dims(s_matrix)

            return s_matrix #Add one more dimension #(None, MAX_PASSAGE_PER_Q, MAX_Q_TERM, #(None, MAX_PASSAGE_PER_Q, MAX_Q_TERM, EMB_DIM, 1)
        elif self.interaction_mode==1:
            raise NotImplementedError("interaction mode of layer SimilarityMatrix is not implemented")

            
class MaskedConv2D(Layer):
    
    def __init__(self, filters, kernel_size, activation, regularizer=None, **kargs):
        super(MaskedConv2D, self).__init__(**kargs)

        self.activation = activations.get(activation)
        
        if regularizer is None or isinstance(regularizer,str):
            self.regularizer = regularizers.get(regularizer)
        else:
            self.regularizer = regularizer
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation

    def build(self, input_shape):

        self.conv2dlayer = Conv2D( filters = self.filters, kernel_size=self.kernel_size, activation=self.activation, kernel_regularizer=self.regularizer )
        self.conv2dlayer.build(input_shape)
        self._trainable_weights = self.conv2dlayer.trainable_weights
        
        super(MaskedConv2D, self).build(input_shape)
    
    def call(self, x):
        
        condition = K.all(x) #if all the values are the same
        inv_condition = (1-K.cast(condition, K.floatx()))
        print(inv_condition)
        feature_maps = self.conv2dlayer(x)
        
        return feature_maps * inv_condition

    
class MultipleMaskedConv2D(Layer):
    
    def __init__(self, filters, kernel_size, activation, initializer='glorot_normal', regularizer=None, **kargs):
        super(MaskedConv2D, self).__init__(**kargs)

        self.activation = activations.get(activation)
        self.initializer = initializers.get(initializer)
        
        if regularizer is None or isinstance(regularizer,str):
            self.regularizer = regularizers.get(regularizer)
        else:
            self.regularizer = regularizer
        
        self.filters = filters
        self.kernel_size = kernel_size

    def build(self, input_shape):

        input_filter = int(input_shape[-1])
        
        self.kernel_3_3 = self.add_variable(name = "conv_kernel_3_3",
                                   shape = (3,3,input_filter,CNN_FILTERS),
                                   initializer = self.initializer,
                                   regularizer = self.regularizer,)
        
        self.kernel_5_1 = self.add_variable(name = "conv_kernel_5_1",
                                   shape = (5,1,input_filter,CNN_FILTERS),
                                   initializer = self.initializer,
                                   regularizer = self.regularizer,)
        
        self.kernel_1_5 = self.add_variable(name = "conv_kernel_1_5",
                                   shape = (1,5,input_filter,CNN_FILTERS),
                                   initializer = self.initializer,
                                   regularizer = self.regularizer,)
        
        self.kernel_3_3_bias = self.add_variable(name = "conv_kernel_3_3_bias",
                                   shape = (self.filters,),)
        
        self.kernel_5_1_bias = self.add_variable(name = "conv_kernel_5_1_bias",
                                   shape = (self.filters,),)
        
        self.kernel_1_5_bias = self.add_variable(name = "conv_kernel_1_5_bias",
                                   shape = (self.filters,),)
        
        #end dimensions = 7, 9, 100

        
        super(MaskedConv2D, self).build(input_shape)
    
    def call(self, x):
        
        condition = K.all(x) #if all the values are the same
        inv_condition = (1-K.cast(condition, K.floatx()))
        
        kernel_3_3 = K.conv2d(x, self.kernel_3_3)
        kernel_3_3 = K.bias_add(kernel_3_3, self.kernel_3_3_bias)
        kernel_3_3 = self.activation(kernel_3_3)
        kernel_3_3_pool = K.pool2d(kernel_3_3,(11,13))
        
        kernel_5_1 = K.conv2d(x, self.kernel_5_1)
        kernel_5_1 = K.bias_add(kernel_5_1, self.kernel_5_1_bias)
        kernel_5_1 = self.activation(kernel_5_1)
        kernel_5_1_pool = K.pool2d(kernel_5_1,(9,15))
        
        kernel_1_5 = K.conv2d(x, self.kernel_1_5)
        kernel_1_5 = K.bias_add(kernel_1_5, self.kernel_1_5_bias)
        kernel_1_5 = self.activation(kernel_1_5)
        kernel_1_5_pool = K.pool2d(kernel_1_5,(13,11))
        
        print(kernel_3_3_pool)
        print(kernel_5_1_pool)
        print(kernel_1_5_pool)
        
        kernel_3_3_flat = K.reshape(kernel_3_3_pool,(-1,self.filters))
        kernel_5_1_flat = K.reshape(kernel_5_1_pool,(-1,self.filters))
        kernel_1_5_flat = K.reshape(kernel_1_5_pool,(-1,self.filters))
        print(kernel_3_3_flat)
        print(kernel_5_1_flat)
        print(kernel_1_5_flat)
        
        concat =  K.concatenate([kernel_3_3_flat,kernel_5_1_flat,kernel_1_5_flat])
        #print(concat)
        
        #proj = K.dot(concat, self.dense)
        #proj = K.bias_add(proj,self.dense_bias)
        #proj = self.activation(proj)

        
        return concat * inv_condition
    
class TermGatingDRMM_FFN(Layer):
    
    def __init__(self, embedding_dim, rnn_dim, activation=None, initializer='glorot_normal', regularizer=None):
        super(TermGatingDRMM_FFN, self).__init__()

        self.activation = activations.get(activation)
        self.initializer = initializers.get(initializer)
        
        if regularizer is None or isinstance(regularizer,str):
            self.regularizer = regularizers.get(regularizer)
        else:
            self.regularizer = regularizer
        
        self.emb_dim = embedding_dim
        self.rnn_dim = rnn_dim

    def build(self, input_shape):
        
        #term gating W
        self.W_query = self.add_variable(name = "term_gating_We",
                                   shape = [self.emb_dim,1],
                                   initializer = self.initializer,
                                   regularizer = self.regularizer,)
        
        self.dense_score = Dense(1,kernel_regularizer = self.regularizer, activation=self.activation)
        
        dense_shape = input_shape[1]
        print(dense_shape)
        
        self.dense_score.build((dense_shape[0],dense_shape[2]))
        self._trainable_weights += self.dense_score.trainable_weights
        #self.ones = K.constant(np.ones((aggreation_dimension,1)))
        
        super(TermGatingDRMM_FFN, self).build(input_shape)
    
    def call(self, x):
        
        query_embeddings = x[0] #(None, MAX_Q_TERM, EMB_SIZE)
        snippet_representation_per_query = x[1] #(None, MAX_Q_TERM, BI_GRU_DIM)
        
        #compute gated weights
        gated_logits = K.squeeze(K.dot(query_embeddings, self.W_query), axis = -1 )
        #print(gated_logits)
        gated_distribution = K.expand_dims(K.softmax(gated_logits))
        #print(gated_distribution)
        #snippet projection
        self.attention_weights = gated_distribution
        
        weighted_score = K.sum(snippet_representation_per_query * gated_distribution,  axis = 1)
        print(weighted_score)
        
        return self.dense_score(weighted_score) # Replace with K.sum of all elements?
    

class SelfAttention(Layer):
    
    def __init__(self, attention_dimension, initializer='glorot_normal', regularizer=None, **kargs):
        super(SelfAttention, self).__init__(**kargs)

        self.initializer = initializer
        self.attention_dimension = attention_dimension
        
        self.num_of_self_attention = 1
        
        if regularizer is None or isinstance(regularizer,str):
            self.regularizer = regularizers.get(regularizer)
        else:
            self.regularizer = regularizer

    def build(self, input_shape):
        
        emb_dim = int(input_shape[2])

        self.W_attn_project = self.add_variable(name = "self_attention_projection",
                                   shape = [emb_dim, self.attention_dimension],
                                   initializer = self.initializer,
                                   regularizer = self.regularizer,)
        
        self.W_attn_score = self.add_variable(name = "self_attention_score",
                                   shape = [self.attention_dimension, self.num_of_self_attention],
                                   initializer = self.initializer,
                                   regularizer = self.regularizer,)

        super(SelfAttention, self).build(input_shape)
    
    def call(self, x):
        
        #x_transpose = K.permute_dimensions(x,[0,2,1]) # (NONE, 300, 15)
        #print("x_transpose",x_transpose)
        x_projection = K.dot(x, self.W_attn_project) # (NONE, 300, 300)
        print("x_projection",x_projection)
        x_tanh = K.tanh(x_projection) # (NONE, 300, 15)
        print("x_tanh",x_tanh)
        x_attention = K.dot(x_tanh, self.W_attn_score)
        print("x_attention",x_attention)
        #x_attention_squeeze = K.squeeze(x_score, axis=1)
        x_attention_softmax = K.softmax(x_attention,axis = 1)
        print("x_attention_softmax",x_attention_softmax)
        
        if (self.num_of_self_attention>1):
            x_attention_softmax_transpose =  K.permute_dimensions(x_attention_softmax,[0,2,1])
            print("x_attention_softmax_transpose",x_attention_softmax_transpose)

            x_attention_softmax_transpose_expand = K.expand_dims(x_attention_softmax_transpose)
            print("x_attention_softmax_transpose_expand",x_attention_softmax_transpose_expand)

            x_scored_emb = x_attention_softmax_transpose_expand * x
            print("x_scored_emb",x_scored_emb)
            x_attention_rep = K.sum(x_scored_emb, axis=2)
        else:
            x_scored_emb = x_attention_softmax * x
            print("x_scored_emb",x_scored_emb)
            x_attention_rep = K.sum(x_scored_emb, axis=1)
            
        print("x_attention_rep",x_attention_rep)
        return x_attention_rep
    
    
class MaskedSelfAttention(Layer):
    
    def __init__(self, attention_dimension, initializer='glorot_normal', regularizer=None, **kargs):
        super(MaskedSelfAttention, self).__init__(**kargs)

        self.initializer = initializer
        self.attention_dimension = attention_dimension
        
        self.num_of_self_attention = 1
        self.attention_weights = []
        
        if regularizer is None or isinstance(regularizer,str):
            self.regularizer = regularizers.get(regularizer)
        else:
            self.regularizer = regularizer

    def build(self, input_shape):
        
        emb_dim = int(input_shape[2])

        self.W_attn_project = self.add_variable(name = "self_attention_projection",
                                   shape = [emb_dim, self.attention_dimension],
                                   initializer = self.initializer,
                                   regularizer = self.regularizer,)
        
        self.W_attn_score = self.add_variable(name = "self_attention_score",
                                   shape = [self.attention_dimension, self.num_of_self_attention],
                                   initializer = self.initializer,
                                   regularizer = self.regularizer,)

        super(MaskedSelfAttention, self).build(input_shape)
    
    def call(self, x):
        
        #x_transpose = K.permute_dimensions(x,[0,2,1]) # (NONE, 100,15)
        #print("x_transpose",x_transpose)
        condition = K.all(x,keepdims=True,axis=-1)
        print("condition",condition)
        inv_condition = (1-K.cast(condition, K.floatx()))
        print("inv_condition",inv_condition)
        
        x_projection = K.dot(x, self.W_attn_project) # (NONE, 300, 300)
        print("x_projection",x_projection)
        x_tanh = K.tanh(x_projection) # (NONE, 300, 15)
        print("x_tanh",x_tanh)
        x_attention = K.dot(x_tanh, self.W_attn_score)
        print("x_attention",x_attention)
        x_attention_maked = x_attention + (1.0 - inv_condition) * -10000.0
        print("x_attention_maked",x_attention_maked)
        #x_attention_squeeze = K.squeeze(x_score, axis=1)
        x_attention_softmax = K.softmax(x_attention_maked,axis = 1)
        print("x_attention_softmax",x_attention_softmax)
        
        if (self.num_of_self_attention>1):
            x_attention_softmax_transpose =  K.permute_dimensions(x_attention_softmax,[0,2,1])
            print("x_attention_softmax_transpose",x_attention_softmax_transpose)

            x_attention_softmax_transpose_expand = K.expand_dims(x_attention_softmax_transpose)
            print("x_attention_softmax_transpose_expand",x_attention_softmax_transpose_expand)

            x_scored_emb = x_attention_softmax_transpose_expand * x
            print("x_scored_emb",x_scored_emb)
            x_attention_rep = K.sum(x_scored_emb, axis=2)
        else:
            x_scored_emb = x_attention_softmax * x
            print("x_scored_emb",x_scored_emb)
            x_attention_rep = K.sum(x_scored_emb, axis=1)
        
        #save attent w
        self.attention_weights.append(x_attention_softmax)
        
        print("x_attention_rep",x_attention_rep)
        return x_attention_rep

    
class FullMaskedSelfAttention(Layer):
    
    def __init__(self, attention_dimension, initializer='glorot_normal', regularizer=None, **kargs):
        super(FullMaskedSelfAttention, self).__init__(**kargs)

        self.initializer = initializer
        self.attention_dimension = attention_dimension
        
        self.num_of_self_attention = 1
        self.attention_weights = []
        
        if regularizer is None or isinstance(regularizer,str):
            self.regularizer = regularizers.get(regularizer)
        else:
            self.regularizer = regularizer

    def build(self, input_shape):
        
        emb_dim = int(input_shape[2])

        self.W_attn_project = self.add_variable(name = "self_attention_projection",
                                   shape = [emb_dim, self.attention_dimension],
                                   initializer = self.initializer,
                                   regularizer = self.regularizer,)
        
        self.W_attn_score = self.add_variable(name = "self_attention_score",
                                   shape = [self.attention_dimension, self.num_of_self_attention],
                                   initializer = self.initializer,
                                   regularizer = self.regularizer,)

        super(FullMaskedSelfAttention, self).build(input_shape)
    
    def call(self, x):
        
        #x_transpose = K.permute_dimensions(x,[0,2,1]) # (NONE, 100,15)
        #print("x_transpose",x_transpose)
        condition = K.all(x,keepdims=True,axis=-1)
        print("condition",condition)
        inv_condition = (1-K.cast(condition, K.floatx()))
        print("inv_condition",inv_condition)
        
        condition_all_zero = K.all(x,keepdims=True)
        print("condition_all_zero",condition_all_zero)
        inv_condition_all_zero = (1-K.cast(condition_all_zero, K.floatx()))
        print("inv_condition_all_zero",inv_condition_all_zero)
        
        x_projection = K.dot(x, self.W_attn_project) # (NONE, 300, 300)
        print("x_projection",x_projection)
        x_tanh = K.tanh(x_projection) # (NONE, 300, 15)
        print("x_tanh",x_tanh)
        x_attention = K.dot(x_tanh, self.W_attn_score)
        print("x_attention",x_attention)
        x_attention_maked = x_attention + (1.0 - inv_condition) * -10000.0
        print("x_attention_maked",x_attention_maked)
        #x_attention_squeeze = K.squeeze(x_score, axis=1)
        x_attention_softmax = K.softmax(x_attention_maked,axis = 1)
        print("x_attention_softmax",x_attention_softmax)
        x_attention_softmax_masked = x_attention_softmax * inv_condition_all_zero
        print("x_attention_softmax_masked",x_attention_softmax_masked)
        
        if (self.num_of_self_attention>1):
            x_attention_softmax_transpose =  K.permute_dimensions(x_attention_softmax_masked,[0,2,1])
            print("x_attention_softmax_transpose",x_attention_softmax_transpose)

            x_attention_softmax_transpose_expand = K.expand_dims(x_attention_softmax_transpose)
            print("x_attention_softmax_transpose_expand",x_attention_softmax_transpose_expand)

            x_scored_emb = x_attention_softmax_transpose_expand * x
            print("x_scored_emb",x_scored_emb)
            x_attention_rep = K.sum(x_scored_emb, axis=2)
        else:
            x_scored_emb = x_attention_softmax_masked * x
            print("x_scored_emb",x_scored_emb)
            x_attention_rep = K.sum(x_scored_emb, axis=1)
        
        #save attent w
        self.attention_weights.append(x_attention_softmax_masked)
        
        print("x_attention_rep",x_attention_rep)
        return x_attention_rep
    
class CrossAttention(Layer):
    
    def __init__(self, initializer='glorot_normal', regularizer=None, **kargs):
        super(CrossAttention, self).__init__(**kargs)

        self.initializer = initializer
        
        if regularizer is None or isinstance(regularizer,str):
            self.regularizer = regularizers.get(regularizer)
        else:
            self.regularizer = regularizer

    def build(self, input_shape):
        
        """
        input: [0] - query context embedding
               [1] - document context embedding
        """
        doc_embedding = input_shape[1]
        query_embedding = input_shape[0]
        
        self.query_len = query_embedding[1]
        self.doc_len = doc_embedding[1]
        
        print("query_len", self.query_len)
        print("doc_len", self.doc_len)
        
        assert int(query_embedding[2]) == int(doc_embedding[2])
        
        self.embedding_dim = int(query_embedding[2])
        
        self.W_sim_projection = self.add_variable(name = "similarity_projection",
                                   shape = [self.embedding_dim*3,1],
                                   initializer = self.initializer,
                                   regularizer = self.regularizer,)
        
        super(CrossAttention, self).build(input_shape)
    
    def call(self, x):
        """
        input: [0] - query context embedding
               [1] - document context embedding
        """
        doc_embedding = x[1]
        print("doc_embedding",doc_embedding.shape)
        query_embedding = x[0]
        print("query_embedding",query_embedding.shape)
        
        #build similarity matrix
        #row document token
        #colum query token
        doc_q_matrix = K.expand_dims(doc_embedding, axis=2)
        print("doc_q_matrix",doc_q_matrix.shape)
        doc_q_matrix = K.repeat_elements(doc_q_matrix, self.query_len, axis=2)
        print("doc_q_matrix",doc_q_matrix.shape)
        
        q_doc_matrix = K.expand_dims(query_embedding, axis=1)
        print("q_doc_matrix",q_doc_matrix.shape)
        q_doc_matrix = K.repeat_elements(q_doc_matrix, self.doc_len, axis=1)
        print("q_doc_matrix",q_doc_matrix.shape)
        
        element_mult = doc_q_matrix * q_doc_matrix
        print("element_mult",element_mult.shape)
        
        #concatenation
        S = K.concatenate([doc_q_matrix, q_doc_matrix, element_mult])
        print("S",S.shape)
        print("Wc",self.W_sim_projection.shape)
        S = K.dot(S, self.W_sim_projection)
        print("S",S.shape)
        
        S = K.squeeze(S,axis=-1)
        print("S",S.shape)
        
        S_D2Q = K.softmax(S, axis=1)
        print("S_D2Q",S_D2Q.shape)
        
        S_Q2D = K.softmax(S, axis=2)
        print("S_Q2D",S_Q2D.shape)
        
        A_D2Q = K.batch_dot(S_D2Q, query_embedding)
        print("A_D2Q",A_D2Q.shape)
        
        S_Q2D_transpose = K.permute_dimensions(S_Q2D,[0,2,1])
        print("S_Q2D_transpose",S_Q2D_transpose.shape)
        
        A_D2Q_Q2D = K.batch_dot(S_D2Q, S_Q2D_transpose)
        print("A_D2Q_Q2D",A_D2Q_Q2D.shape)
        
        A_Q2D = K.batch_dot(A_D2Q_Q2D, doc_embedding)
        print("A_Q2D",A_Q2D.shape)
        
        #concat
        doc_attn = doc_embedding * A_D2Q
        print("doc_attn",doc_attn.shape)
        doc_q_attn = doc_embedding * A_Q2D
        print("doc_q_attn",doc_q_attn.shape)
        
        V = K.concatenate([doc_embedding, A_D2Q, doc_attn, doc_q_attn])
        
        return V
