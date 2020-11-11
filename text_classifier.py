import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from dataset import X_train, X_test, y_train, y_test
from dataset import num_classes, maxlen, num_chars

hdims = 128
kl_loss = False

class KLLossLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(KLLossLayer, self).__init__(**kwargs)

    def call(self, inputs):
        mu = tf.reduce_mean(inputs, axis=1, keepdims=True)
        var = tf.reduce_sum(tf.square(inputs - mu), axis=1) / (hdims-1)
        kl_loss = 0.5 * tf.reduce_mean(-tf.math.log(var) + tf.square(mu) + var - 1)
        self.add_loss(kl_loss)
        return inputs

class MaskGlobalMaxPooling1D(tf.keras.layers.Layer):
    
    def __init__(self, **kwargs):
        super(MaskGlobalMaxPooling1D, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        if mask is None:
            mask = 1
        else:
            # 扩展维度便于广播
            mask = tf.expand_dims(tf.cast(mask, "float32"), -1)
        x = inputs
        x = x - (1 - mask) * 1e12 # 用一个大的负数mask
        return tf.reduce_max(x, axis=1)

inputs = Input(shape=(maxlen,))
mask = Lambda(lambda x: tf.not_equal(x, 0))(inputs)
embedding = Embedding(num_chars, hdims, embeddings_initializer="uniform", mask_zero=True)

x = embedding(inputs)
x = Conv1D(filters=hdims, kernel_size=2, padding="same", activation="relu")(x)
x = MaskGlobalMaxPooling1D()(x, mask=mask)

if kl_loss:
    x = KLLossLayer()(x)

outputs = Dense(num_classes, activation="softmax")(x)
model = Model(inputs, outputs)
model.summary()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(X_train, y_train, shuffle=True, batch_size=32, epochs=10, validation_data=(X_test, y_test))