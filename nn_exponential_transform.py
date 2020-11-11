import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

hdims = 1280

def gelu(x):
    return 0.5 * x * (1.0 + tf.math.erf(x / tf.sqrt(2.0)))

class CentralLoss(tf.keras.losses.Loss):
	"""有利于控制分布中心"""

	def __init__(self, **kwargs):
		super(CentralLoss, self).__init__(**kwargs)

	def call(self, y_true, y_pred):
		mu1 = tf.reduce_mean(y_true, axis=1, keepdims=True)
		mu2 = tf.reduce_mean(y_pred, axis=1, keepdims=True)
		loss = tf.reduce_mean(tf.square(1.0 / mu1 - 1.0 / mu2))
		return loss

class KLLossLayer(tf.keras.layers.Layer):
	"""
	\log {\frac {\lambda _{1}}{\lambda_{2} }}+{\frac {\lambda_{2} }{\lambda _{1}}}-1
	"""

	def __init__(self, scale=3, **kwargs):
		super(KLLossLayer, self).__init__(**kwargs)
		self.scale = scale

	def call(self, inputs):
		mu = tf.reduce_mean(inputs, axis=1, keepdims=False)
		kl_loss = tf.reduce_mean(tf.math.log(self.scale / mu) + mu / self.scale - 1)
		self.add_loss(kl_loss)
		return inputs

inputs = tf.keras.layers.Input(shape=(hdims,))
x = inputs
x = tf.keras.layers.Dense(hdims, activation="relu")(x)
x = tf.keras.layers.Dense(hdims)(x)
x = KLLossLayer()(x)
outputs = x
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer="adam")#, loss=CentralLoss())
model.summary()


X = np.random.uniform(0.001, 5, size=(32*200, hdims))
y = np.random.exponential(scale=3, size=(32*200, hdims))

model.fit(X, y, epochs=10, batch_size=32)

def test(x=None):
	if x is None:
		x = np.random.uniform(0.001, 1, size=(1, hdims))
	y = model.predict(x)

	plt.subplot(211)
	plt.hist(x[0], bins=100, density=True)
	plt.subplot(212)
	plt.hist(y[0], bins=100, density=True)

	mu = np.mean(y[0])
	x = np.linspace(0.001, 5 * mu, hdims)
	y = stats.expon.pdf(x, scale=mu)
	plt.plot(x, y, 'b')

	# mu = 0
	# sigma = 1
	# x = np.linspace(mu - 4*sigma, mu + 4*sigma, hdims)
	# y = stats.norm.pdf(x, mu, sigma)
	# plt.plot(x, y, 'r')

	plt.show()

test(np.random.uniform(0.001, 1, size=(1, hdims)))
test(np.random.uniform(0.001, 1, size=(1, hdims)))
test(np.random.uniform(0.001, 1, size=(1, hdims)))
