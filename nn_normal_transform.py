import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

hdims = 1280

def gelu(x):
    return 0.5 * x * (1.0 + tf.math.erf(x / tf.sqrt(2.0)))

class CentralLoss(tf.keras.losses.Loss):
	"""有利于控制分布中心"""

	def __init__(self, std=False, **kwargs):
		super(CentralLoss, self).__init__(**kwargs)
		self.std = std

	def call(self, y_true, y_pred):
		mu1 = tf.reduce_mean(y_true, axis=1, keepdims=True)
		mu2 = tf.reduce_mean(y_pred, axis=1, keepdims=True)
		loss1 = tf.reduce_mean(tf.square(mu1 - mu2))
		if self.std:
			s1 = tf.reduce_mean(y_true - mu1)
			s2 = tf.reduce_mean(y_pred - mu2)
			loss2 = tf.reduce_mean(tf.square(s1 - s2))
		else:
			loss2 = 0
		return loss1 + loss2

class KLLossLayer(tf.keras.layers.Layer):

	def __init__(self, **kwargs):
		super(KLLossLayer, self).__init__(**kwargs)

	def call(self, inputs):
		mu = tf.reduce_mean(inputs, axis=1, keepdims=True)
		var = tf.reduce_sum(tf.square(inputs - mu), axis=1) / (hdims-1)
		kl_loss = 0.5 * tf.reduce_mean(-tf.math.log(var) + tf.square(mu) + var - 1)
		self.add_loss(kl_loss)
		return inputs

inputs = tf.keras.layers.Input(shape=(hdims,))
x = inputs
x = tf.keras.layers.Dense(hdims, activation="tanh")(x)
x = tf.keras.layers.Dense(hdims)(x)
x = KLLossLayer()(x)
outputs = x
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer="adam", loss=CentralLoss(std=False))
model.summary()


X = np.random.uniform(0, 1, size=(32*200, hdims))
y = np.random.normal(size=(32*200, hdims))

model.fit(X, y, epochs=10, batch_size=32)

def test(x=None):
	if x is None:
		x = np.random.uniform(0, 1, size=(1, hdims))
	y = model.predict(x)

	plt.subplot(211)
	plt.hist(x[0], bins=100, density=True)
	plt.subplot(212)
	plt.hist(y[0], bins=100, density=True)

	mu = np.mean(y[0])
	sigma = np.std(y[0])
	x = np.linspace(mu - 4*sigma, mu + 4*sigma, hdims)
	y = stats.norm.pdf(x, mu, sigma)
	plt.plot(x, y, 'b')

	mu = 0
	sigma = 1
	x = np.linspace(mu - 4*sigma, mu + 4*sigma, hdims)
	y = stats.norm.pdf(x, mu, sigma)
	plt.plot(x, y, 'r')

	plt.show()

test(np.random.uniform(0, 1, size=(1, hdims)))
test(np.random.uniform(0, 1, size=(1, hdims)))
test(np.random.uniform(0, 1, size=(1, hdims)))
