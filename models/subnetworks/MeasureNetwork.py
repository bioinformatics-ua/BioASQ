from tensorflow.keras.layers import Conv2D, Lambda, GlobalMaxPooling2D, TimeDistributed, Concatenate, Bidirectional, GRU
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K
from tensorflow import unstack
from logger import log


class MeasureNetwork(Model):
    def __init__(self, Q, S, P, gru_bidirectional, gru_dim, filters, kernel, regularizer, **kwargs):
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
        if gru_bidirectional:
            self.gru = Bidirectional(GRU(gru_dim, kernel_regularizer=regularizer, name="gru_passage_aggregation", **kwargs))
        else:
            self.gru = GRU(gru_dim, kernel_regularizer=regularizer, name="gru_passage_aggregation", **kwargs)
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
            gru_representation = self.gru(concatenation)
            relevance_representation.append(self.add_passage_dimension(gru_representation))

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
