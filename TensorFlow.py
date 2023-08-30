import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Convert images and labels to tensors
train_images = tf.convert_to_tensor(train_images, dtype=tf.float32)
train_labels = tf.convert_to_tensor(train_labels, dtype=tf.int32)
test_images = tf.convert_to_tensor(test_images, dtype=tf.float32)
test_labels = tf.convert_to_tensor(test_labels, dtype=tf.int32)

batch_size = 64

# Create a TensorFlow dataset from the data
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

# Shuffle and batch the training dataset
batch_size = 64
train_dataset = train_dataset.shuffle(buffer_size=len(train_images)).batch(batch_size)


# for X, y in test_dataset:
#     print(f"Shape of X [N, C, H, W]: {X.shape}")
#     print(f"Shape of y: {y.shape} {y.dtype}")
#     break

# Choose CPU or GPU to execute the code
devices = tf.config.list_physical_devices("GPU")
if len(devices) > 0:
    device = "GPU"
else:
    device = "CPU"

print(f"Using {device} device")

# Neural network 
# Don't need to do the low-level details, keras can do it
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28,28)),
  tf.keras.layers.Dense(512, activation='sigmoid'),
  tf.keras.layers.Dense(256, activation='sigmoid'),
  tf.keras.layers.Dense(10)
])

# Choose the optimizer algorithm, loss function, and evaluation metrics
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()], #the model will compute the accuracy using SparseCategoricalAccuracy()
)

# Training (and maybe testing) process
num_epochs=5
model.fit(
    train_dataset,
    epochs=num_epochs,
    batch_size=batch_size,
)
    
# test_dataset_batched = test_dataset.batch(batch_size)
# test_loss, test_accuracy = model.evaluate(test_dataset_batched)
# print(f"Test Loss: {test_loss}")
# print(f"Test Accuracy: {test_accuracy")

print("Done!")


#60000 samples/64 (batch size) = 938 batches