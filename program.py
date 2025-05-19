from modules.semantic_ids import SemanticIds
from modules.loss import rq_vae_loss
from modules.train import train_step
import tensorflow as tf


if __name__ == "__main__":
    # Create a dummy dataset
    x = tf.random.normal((32, 512))  # Batch of 32 samples, each with 512 features

    # Initialize the model
    model = SemanticIds(num_layers=3, codebook_size=[256, 256, 256], codebook_dim=512)
    optimizer = tf.keras.optimizers.Adam()

    # Train the model
    loss_fn = rq_vae_loss
    for epoch in range(10):
        loss = train_step(model, optimizer, x, loss_fn)
        print(f"Epoch {epoch + 1}, Loss: {loss.numpy()}")