import tensorflow as tf
from rq_vae import RqVae
from typing import Union, List

class SemanticIds(tf.keras.layers.Layer):
    def __init__(self, num_layers: int, codebook_size: Union[List, List[int]], codebook_dim: int = 512):
        super(SemanticIds, self).__init__()
        self.encoder = tf.keras.layers.Dense(codebook_dim)
        self.semantic_id_generator = SemanticIdGenerator(num_layers, codebook_size, codebook_dim)
        self.decoder = tf.keras.layers.Dense(codebook_dim)

    def call(self, dense_content_embedding: tf.Tensor, training=True) -> tf.Tensor:
        r_0 = self.encoder(dense_content_embedding)
        if training:
            reconstructed_latent_vector = self.semantic_id_generator(r_0)
            return self.decoder(reconstructed_latent_vector)
        else:
            return self.semantic_id_generator(r_0)


class SemanticIdGenerator(tf.keras.layers.Layer):
    def __init__(self, num_layers: int, codebook_size: Union[List, List[int]], codebook_dim: int = 512):
        super(SemanticIdGenerator, self).__init__()
        if isinstance(codebook_size, int):
            codebook_size = [codebook_size] * num_layers
        elif len(codebook_size) != num_layers:
            raise ValueError("codebook_size must be an int or a list of length num_layers")
        self.rq_vae_layers = [RqVae(codebook_size[i], codebook_dim) for i in range(num_layers)]

    def call(self, dense_content_embedding: tf.Tensor, training=True) -> tf.Tensor:
        residual = dense_content_embedding
        if training:
            reconstruction = []
            for rq_vae in self.rq_vae_layers:
                rq_vae_output = rq_vae(residual)
                reconstruction.append(rq_vae_output[1])
                residual = rq_vae_output[1]
            return tf.reduce_sum(tf.stack(reconstruction, axis=-1), axis=-1)
        else:
            layer_indices = []
            for rq_vae in self.rq_vae_layers:
                rq_vae_output = rq_vae(residual)
                layer_indices.append(rq_vae_output[0])
                residual = rq_vae_output[1]
            return tf.stack(layer_indices, axis=1)
