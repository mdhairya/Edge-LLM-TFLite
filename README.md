# Edge-LLM-TFLite

A demonstration of building, quantizing, and deploying a miniature Language Model (LLM) for edge devices (Android, Raspberry Pi, IoT) using advanced TensorFlow and TensorFlow Lite.

## Overview
This repository contains a full pipeline:
1. **Model Definition**: A custom miniature Transformer architecture using TensorFlow/Keras Subclassing.
2. **Export**: Saving the model as a TensorFlow `SavedModel`.
3. **Quantization**: Converting the model to `.tflite` format using Post-Training Quantization (PTQ) with a Representative Dataset to achieve Full Integer (INT8) Quantization.
4. **Inference**: Loading the `.tflite` model via `tf.lite.Interpreter` for fast, low-memory on-device execution.

## Getting Started

### Prerequisites
Install the required dependencies:
```bash
pip install -r requirements.txt
