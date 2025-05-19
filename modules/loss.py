import tensorflow as tf

def rq_vae_loss(x, z, x_recon, codebook_embeddings, beta=0.25):
    """
    Compute the loss for the RQ-VAE model.
    
    Args:
        z: The original latent vector.
        z_recon: The reconstructed latent vector.
        codebook_embeddings: The codebook embeddings.
        codebook_size: The size of the codebook.
        beta: The weight for the reconstruction loss.
        
    Returns:
        The computed loss value.
    """
    recon_loss = tf.reduce_mean(tf.square(z - x_recon))
    # codebook_embeddings are shape (None, codebook_dim, num_layers)
    # z is shape (None, codebook_dim)
    num_layers = tf.shape(codebook_embeddings)[-1]
    z_tiled = tf.tile(tf.expand_dims(z, axis=-1), [1, 1, num_layers])
    distances = tf.reduce_sum(tf.square(z_tiled - codebook_embeddings), axis=(1, 2))
    codebook_loss = tf.reduce_mean(distances)
    total_loss = recon_loss + beta * codebook_loss
    return total_loss