import tensorflow as tf
from typing import NamedTuple


class RqVaeOutput:
    closest_codebook_index: tf.Tensor
    next_residual: tf.Tensor

class RqVae(tf.keras.layers.Layer):
    def __init__(self, codebook_size, residual_dimension):
        super(SemanticIds, self).__init__()
        self.codebook_embeddings = tf.Variable(shape=(codebook_size, residual_dimension), dtype=tf.float32)


    def call(self, previous_residual):
        distances = tf.reduce_sum(tf.square(previous_residual[:, tf.newaxis, :] - self.codebook_embeddings[tf.newaxis, :, :]), axis=-1)
        closest_codebook_index = tf.argmin(distances, axis=1)
        next_residual = previous_residual - codebook_embedding
        return RqVaeOutput(codebook_embedding_index=closest_codebook_index, next_residual=next_residual)


        