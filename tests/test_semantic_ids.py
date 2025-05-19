from modules.semantic_ids import SemanticIdGenerator
from modules.rq_vae import RqVae
import tensorflow as tf

def test_rqvae():
    codebook_size = 256
    codebook_dim = 512
    rq_vae = RqVae(codebook_size=codebook_size, residual_dimension=codebook_dim)
    constant_variable = tf.range(0, codebook_size, dtype=tf.float32)
    constant_variable_tiled = tf.tile(tf.expand_dims(constant_variable, axis=-1), [1, codebook_dim])
    rq_vae.codebook_embeddings = tf.Variable(initial_value=constant_variable_tiled, dtype=tf.float32)

    previous_residual = tf.constant([[7.0] * codebook_dim])
    closest_codebook_index, next_residual, codebook_embedding = rq_vae(previous_residual)
    assert closest_codebook_index.shape == (1,)
    assert closest_codebook_index.numpy()[0] == 7

def test_semantic_ids():
    num_layers = 2
    codebook_size = [256] * num_layers
    codebook_dim = 512
    semantic_id_gen = SemanticIdGenerator(num_layers=num_layers, codebook_size=codebook_size, codebook_dim=codebook_dim)
    default_embedding = [[9999] * codebook_dim for _ in range(codebook_size[0])]
    rqvae1 = default_embedding.copy()

    rqvae1[0] = ([0] * (codebook_dim // 2)) + ([1] * (codebook_dim // 2))
    rqvae2 = default_embedding.copy()
    rqvae2[1] = ([1] * (codebook_dim // 2)) + ([0] * (codebook_dim // 2))
    semantic_id_gen.rq_vae_layers[0].codebook_embeddings = tf.Variable(initial_value=rqvae1, dtype=tf.float32)
    semantic_id_gen.rq_vae_layers[1].codebook_embeddings = tf.Variable(initial_value=rqvae2, dtype=tf.float32)
    z = tf.ones((1, codebook_dim))
    zhat_2 = semantic_id_gen(z, training=True)
    tf.debugging.assert_near(tf.reduce_sum(zhat_2, axis=-1), z)

