import time
import configparser
import numpy as np
import tensorflow as tf

def main():
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    tflite_model_path = config['EXPORT']['tflite_model_path']
    maxlen = int(config['MODEL']['max_sequence_length'])
    vocab_size = int(config['MODEL']['vocab_size'])

    print(f"Loading TFLite model from {tflite_model_path}...")
    
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("\n--- TFLite Model Details ---")
    print(f"Input details: {input_details[0]['shape']}, type: {input_details[0]['dtype']}")
    print(f"Output details: {output_details[0]['shape']}, type: {output_details[0]['dtype']}")

    # Create dummy input data (simulating a tokenized text prompt)
    # Shape must match [1, max_sequence_length]
    input_data = np.random.randint(0, vocab_size, size=(1, maxlen), dtype=np.int32)

    # Set the tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    print("\nInvoking Inference on edge simulator...")
    start_time = time.time()
    
    # Run inference
    interpreter.invoke()
    
    end_time = time.time()

    # Extract the output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_token_id = np.argmax(output_data, axis=-1)[0]
    confidence = np.max(output_data, axis=-1)[0]

    print(f"Inference Time: {(end_time - start_time) * 1000:.2f} ms")
    print(f"Predicted Next Token ID: {predicted_token_id} (Confidence: {confidence:.4f})")

if __name__ == "__main__":
    main()
