# %% Setup
import tensorflow as tf
print("TensorFlow version:", tf.__version__)

# %% Loading dataset
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# %% Building model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

# Model returns a vector of logits scores for each class
predictions = model(x_train[:1]).numpy()

# Use softmax function to convert logits to probabilities
tf.nn.softmax(predictions).numpy()

# Define loss function - Loss function takes a vector of logits
# and a True index and returns a scalar loss
# This loss is equal to the negative log probability of the true class:
# The loss is zero if the model is sure of the correct class.
# This untrained model gives probabilities close to random (1/10 for
# each class), so the initial loss should be
# close to -tf.math.log(1/10) ~= 2.3.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], predictions).numpy()

# Compiling model
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# %% Training and evaluating
# Use fit method to adjust model parameters and minimize loss
model.fit(x_train, y_train, epochs=5)

# The evaluate methods checks the model performance, usually on a
# Validation-set or test-set
model.evaluate(x_test,  y_test, verbose=2)

# %% To return probability, wrap model and attach the softmax
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

probability_model(x_test[:5])
