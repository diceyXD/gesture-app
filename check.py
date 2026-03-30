import tensorflow as tf
model = tf.keras.models.load_model('gesture_model.h5', compile=False)
print("INPUT SHAPE:", model.input_shape)
