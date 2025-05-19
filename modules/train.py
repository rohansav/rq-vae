import tensorflow as tf
from modules.loss import rq_vae_loss

@tf.function
def train_step(model, optimizer, x, loss_fn):
    """
    Perform a single training step.
    
    Args:
        model: The model to train.
        optimizer: The optimizer to use.
        data: The input data for the model.
        loss_fn: The loss function to use.
        
    Returns:
        The loss value for the current step.
    """
    with tf.GradientTape() as tape:
        z, zhat_per_layer, xhat = model(x, training=True)
        loss = loss_fn(x, z, xhat, zhat_per_layer)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss
