import configparser
import tensorflow as tf
import numpy as np

def main():
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    saved_model_dir = config['EXPORT']['saved_model_dir']
    tflite_model_path = config['EXPORT']['tflite_model_path']
    maxlen = int(config['MODEL']['max_sequence_length'])
    vocab_size = int(config['MODEL']['vocab_size'])

    print(f"Loading SavedModel from {saved_model_dir}...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    # Enable advanced optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Advanced: Representative Dataset Generator for INT8 Quantization
    # We yield calibration data so TF can calculate activation ranges
    def representative_data_gen():
        for _ in range(100):
            # Generate dummy input matching the input_signature in train.py (1, maxlen)
            data = np.random.randint(0, vocab_size, size=(1, maxlen), dtype=np.int32)
            yield [data]

    converter.representative_dataset = representative_data_gen
    
    # Ensure full integer quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Input/Output tensors must stay as tf.int32 / tf.float32 depending on user needs,
    # but internal ops will be INT8.
    
    print("Converting to Quantized TFLite model...")
    tflite_quant_model = converter.convert()

    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_quant_model)

    print(f"Successfully saved TFLite model to {tflite_model_path}")
    print(f"Model size: {len(tflite_quant_model) / 1024.0:.2f} KB")

if __name__ == "__main__":
    main()
