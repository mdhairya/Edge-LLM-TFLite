import os
import configparser
import numpy as np
import tensorflow as tf
from model import build_mini_llm

def main():
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    # Load hyperparams
    vocab_size = int(config['MODEL']['vocab_size'])
    maxlen = int(config['MODEL']['max_sequence_length'])
    embed_dim = int(config['MODEL']['embed_dim'])
    num_heads = int(config['MODEL']['num_heads'])
    ff_dim = int(config['MODEL']['ff_dim'])
    num_layers = int(config['MODEL']['num_layers'])
    saved_model_dir = config['EXPORT']['saved_model_dir']

    print("Building Miniature LLM...")
    model = build_mini_llm(vocab_size, maxlen, embed_dim, num_heads, ff_dim, num_layers)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(config['TRAINING']['learning_rate'])),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    model.summary()

    # Create dummy dataset for demonstration
    print("Generating dummy training data...")
    num_samples = 1000
    x_train = np.random.randint(0, vocab_size, size=(num_samples, maxlen))
    y_train = np.random.randint(0, vocab_size, size=(num_samples,))

    print("Training model...")
    model.fit(
        x_train, y_train, 
        batch_size=int(config['TRAINING']['batch_size']), 
        epochs=int(config['TRAINING']['epochs']),
        verbose=1
    )

    print(f"Exporting SavedModel to {saved_model_dir}...")
    os.makedirs(saved_model_dir, exist_ok=True)
    # Define a concrete signature for TFLite conversion explicitly
    @tf.function(input_signature=[tf.TensorSpec(shape=[1, maxlen], dtype=tf.int32, name='input_ids')])
    def serving_fn(input_ids):
        return model(input_ids, training=False)

    tf.saved_model.save(model, saved_model_dir, signatures={'serving_default': serving_fn})
    print("Export complete.")

if __name__ == "__main__":
    main()
