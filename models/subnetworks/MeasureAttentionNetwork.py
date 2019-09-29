from tensorflow.keras.layers import Conv2D, Layer, Lambda, GlobalMaxPooling2D, TimeDistributed, Concatenate, Bidirectional, GRU
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K
from tensorflow import unstack
from logger import log


class MeasureNetwork(Model):
    def __init__(self, Q, S, P, attention_dim, filters, kernel, regularizer, **kwargs):
        super().__init__()
        log.info("[MeasureNetwork] kwargs:"+str(kwargs))
        self.Q = Q
        # build cnn extraction patern network
        self.cnn_extraction = Sequential(name="cnn_extraction")
        self.cnn_extraction.add(Masked2DConvLayer(input_shape=(Q, S, 1),
                                                  name="masked_2d_conv",
                                                  filters=filters,
                                                  kernel_size=kernel,
                                                  **kwargs))
        self.cnn_extraction.add(GlobalMaxPooling2D())
        self.cnn_extraction.summary(print_fn=log.info)
        # build propagate cnn_extraction for each passage
        self.pd_cnn_extraction = TimeDistributed(self.cnn_extraction, input_shape=(P, Q, S, 1), name="passage_distributed_cnn_extraction")

        # auxiliar layers
        self.distributed_by_query_term = Lambda(lambda x: unstack(x, axis=1), name="Distributed_by_query_term")
        self.concat_snippet_position = Concatenate(name="concat_snippet_position")
        self.self_attention = MaskedSelfAttention(attention_dim)

        self.add_passage_dimension = Lambda(lambda x: K.expand_dims(x, axis=1), name="add_passage_dimension")
        self.expand_dims_layer = Lambda(lambda x: K.expand_dims(x), name="expand_dims_layer")
        self.reciprocal_function = Lambda(lambda x: 1/(x+2), name="reciprocal_function")
        self.concat_by_passage = Concatenate(axis=1, name="concat_by_passage")

    def call(self, x):
        """
        x[0]: similarity matrix (Q,P,Q,S,1)
        x[1]: snippet position (Q,P,1)
        """
        # Input(shape=())
        similarity_matrix_by_q_term = self.distributed_by_query_term(x[0])
        snippet_position_by_q_term = self.distributed_by_query_term(x[1])

        # same graph for each q term
        relevance_representation = []
        for i in range(self.Q):
            snippet_postion = self.reciprocal_function(snippet_position_by_q_term[i])
            local_relevance = self.pd_cnn_extraction(similarity_matrix_by_q_term[i])
            concatenation = self.concat_snippet_position([local_relevance, self.expand_dims_layer(snippet_postion)])
            self_attention_representation = self.self_attention(concatenation)
            relevance_representation.append(self.add_passage_dimension(self_attention_representation))

        return self.concat_by_passage(relevance_representation)


class Masked2DConvLayer(Conv2D):
    def call(self, x):
        # is a matrix with same value (zero)?
        empty_matrix = K.all(x)
        not_empty_matrix = (1-K.cast(empty_matrix, K.floatx()))
        # normal convolution
        feature_maps = super().call(x)
        # only not empty matrix keep their feature_maps the others are multiplied by zero
        return feature_maps*not_empty_matrix


class MaskedSelfAttention(Layer):

    def __init__(self, attention_dimension, initializer='glorot_normal', regularizer=None, **kargs):
        super(MaskedSelfAttention, self).__init__(**kargs)

        self.initializer = initializer
        self.attention_dimension = attention_dimension

        self.num_of_self_attention = 1
        self.attention_weights = []

        self.regularizer = regularizer

    def build(self, input_shape):

        emb_dim = int(input_shape[2])

        self.W_attn_project = self.add_variable(name="self_attention_projection",
                                                shape=[emb_dim, self.attention_dimension],
                                                initializer=self.initializer,
                                                regularizer=self.regularizer,)

        self.W_attn_score = self.add_variable(name="self_attention_score",
                                              shape=[self.attention_dimension, self.num_of_self_attention],
                                              initializer=self.initializer,
                                              regularizer=self.regularizer,)

        super(MaskedSelfAttention, self).build(input_shape)

    def call(self, x):

        #x_transpose = K.permute_dimensions(x,[0,2,1]) # (NONE, 100,15)
        #print("x_transpose",x_transpose)
        condition = K.all(x, keepdims=True, axis=-1)
        print("condition", condition)
        inv_condition = (1-K.cast(condition, K.floatx()))
        print("inv_condition", inv_condition)

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
