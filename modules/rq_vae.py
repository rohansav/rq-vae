import tensorflow as tf
from typing import NamedTuple


class RqVae(tf.keras.layers.Layer):
    def __init__(self, codebook_size, residual_dimension):
        super(RqVae, self).__init__()
        init = tf.keras.initializers.GlorotUniform()
        self.codebook_embeddings = tf.Variable(
            initial_value=init(shape=(codebook_size, residual_dimension)),
            dtype=tf.float32)


    def call(self, previous_residual):
        distances = tf.reduce_sum(tf.square(previous_residual[:, tf.newaxis, :] - self.codebook_embeddings[tf.newaxis, :, :]), axis=-1)
        closest_codebook_index = tf.argmin(distances, axis=1)
        codebook_embedding = tf.gather(self.codebook_embeddings, closest_codebook_index)
        next_residual = previous_residual - codebook_embedding
        return (closest_codebook_index, next_residual, codebook_embedding)

