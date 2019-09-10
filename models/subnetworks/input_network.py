from tensorflow.keras.layers import Layer, Embedding, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


class DetectionNetwork(Model):
    def __init__(self, Q, P, S, embedding, **kwargs):
        super().__init__()
        self.Q = Q  # number max of query tokens
        self.P = P  # number max of snippets per query token
        self.S = S  # number max of snippet tokens
        self.embedding = embedding

        # build the layers
        self.embedding_layer = Embedding(self.embedding.vocab_size,
                                         self.embedding.embedding_size,
                                         name="embedding_layer",
                                         weights=[self.embedding.embedding_matrix()],
                                         trainable=self.embedding.trainable)

        self.similarity_layer = SimilarityLayer(name="query_snippets_cosine")

        self.auxiliar_transpose_layer = Lambda(lambda x: K.permute_dimensions(x, [0, 1, 2, 4, 3]), name="2D_transpose_in_5th_dimension")

    def call(self, x):
        """
        x[0] is query input with shape=(self.Q,)
        x[1] is snippets input with shape=(self.Q, self.P, self.S)
        """
        query_input = x[0]
        snippets_input = x[1]

        query_embedding = self.embedding_layer(query_input)
        snippet_embedding = self.embedding_layer(snippets_input)
        snippet_embedding = self.auxiliar_transpose_layer(snippet_embedding)
        similarity_matrix = self.similarity_layer([query_embedding, snippet_embedding])

        return similarity_matrix


class SimilarityLayer(Layer):
    def call(self, x):
        query = x[0]
        snippets_per_q_term = x[1]
        # sim => dot product (None, MAX_Q_TERM, EMB_DIM) x (None, MAX_Q_TERM, MAX_PASSAGE_PER_Q, EMB_DIM, QUERY_CENTRIC_CONTEX)
        query = K.expand_dims(query, axis=1)  # (None, 1, MAX_Q_TERM, EMB_DIM)
        query = K.expand_dims(query, axis=1)  # (None, 1, 1, MAX_Q_TERM, EMB_DIM)
        query = K.repeat_elements(query, snippets_per_q_term.shape[1], axis=1)  # ( None, MAX_PASSAGE_PER_Q, MAX_Q_TERM, EMB_DIM)
        query_per_q_term = K.repeat_elements(query, snippets_per_q_term.shape[2], axis=2)
        s_matrix = K.batch_dot(query_per_q_term, snippets_per_q_term)  # ( None, MAX_PASSAGE_PER_Q, MAX_Q_TERM, #(None, MAX_PASSAGE_PER_Q, MAX_Q_TERM, EMB_DIM)
        s_matrix = K.expand_dims(s_matrix)
        return s_matrix
