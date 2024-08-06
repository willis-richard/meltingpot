import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")

# Check if TensorFlow is built with CUDA (GPU) support
print("Is TensorFlow built with CUDA:", tf.test.is_built_with_cuda())

# Check if GPUs are available
gpus = tf.config.list_physical_devices('GPU')
print("Number of GPUs available:", len(gpus))

# Get the name and additional details of the first available GPU
if gpus:
    for gpu in gpus:
       print("GPU details:", gpu)
else:
    print("No GPU detected")
