import tensorflow as tf
from rq_vae import RqVaeOutput, RqVae
from typing import Union, List

class SemanticIds(tf.keras.layers.Layer):
    def __init__(self, num_layers: int, codebook_size: Union[List, List[int]], codebook_dim: int = 512):
        super(SemanticIds, self).__init__()
        self.encoder = tf.keras.layers.Dense(codebook_dim)
        self.semantic_id_generator = SemanticIdGenerator(num_layers, codebook_size, codebook_dim)
        self.decoder = tf.keras.layers.Dense(codebook_dim)

    def call(self, dense_content_embedding: tf.Tensor) -> tf.Tensor:
        r_0 = self.encoder(dense_content_embedding)
        semantic_ids = self.semantic_id_generator(r_0)
        reconstructed_dense_content_embedding = self.decoder(semantic_ids)
        return reconstructed_dense_content_embedding, semantic_ids
class SemanticIdGenerator(tf.keras.layers.Layer):
    def __init__(self, num_layers: int, codebook_size: Union[List, List[int]], codebook_dim: int = 512):
        super(SemanticIds, self).__init__()
        if isinstance(codebook_size, int):
            codebook_size = [codebook_size] * num_layers
        elif len(codebook_size) != num_layers:
            raise ValueError("codebook_size must be an int or a list of length num_layers")
        self.rq_vae_layers = [RqVae(codebook_size[i], 256) for i in range(num_layers)]

    def call(self, dense_content_embedding: tf.Tensor) -> tf.Tensor:
        layer_indices = []
        residual = dense_content_embedding
        for i, rq_vae in enumerate(self.rq_vae_layers):
            rq_vae_output = rq_vae(residual)
            layer_indices.append(rq_vae_output.closest_codebook_index)
            residual = rq_vae_output.next_residual
        return tf.stack(layer_indices, axis=1)


