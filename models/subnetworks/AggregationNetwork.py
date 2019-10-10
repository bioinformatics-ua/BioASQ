from tensorflow.keras.layers import Layer, Lambda, Dense, TimeDistributed, Concatenate, Bidirectional, GRU
from tensorflow.keras import initializers, regularizers, activations
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K
from tensorflow import unstack
from logger import log


class AggregationNetwork(Model):
    def __init__(self, embedding_layer, regularizer, **kwargs):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.term_gating_layer = TermGating(self.embedding_layer.output_dim, regularizer=regularizer, **kwargs)
        self.dense = Dense(1, kernel_regularizer=regularizer, **kwargs)

    def call(self, x):
        """
        x[0]: query input
        x[1]: measure network input
        """
        query_input = x[0]
        measure_input = x[1]
        query_embedding = self.embedding_layer(query_input)
        term_gated = self.term_gating_layer([query_embedding, measure_input])
        return self.dense(term_gated)


class TermGating(Layer):
    def __init__(self, embedding_size, activation=None, initializer='glorot_normal', regularizer=None):
        super().__init__()
        self.embedding_size = embedding_size
        self.activation = activations.get(activation)
        self.initializer = initializers.get(initializer)
        self.regularizer = regularizer
        self.attention_weights = None

    def build(self, input_shape):
        print(input_shape)
        self.W_query = self.add_variable(name="term_gating_We",
                                         shape=[self.embedding_size, 1],
                                         initializer=self.initializer,
                                         regularizer=self.regularizer,)

    def call(self, x):
        """
        x[0]: query embedding
        x[1]: measure network input
        """
        gated_logits = K.squeeze(K.dot(x[0], self.W_query), axis=-1)
        gated_distribution = K.expand_dims(K.softmax(gated_logits))
        self.attention_weights = gated_distribution
        # weighted score
        return K.sum(x[1] * gated_distribution, axis=1)
