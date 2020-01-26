import tensorflow as tf

# Using Python state
x = tf.zeros([10, 10])
x += 2 # This is equivalent to x = x + 2, which does not mutate the original
		# value of x

v = tf.Variable(1.0)
# Use Python's `assert` as a debugging statement to test the condition
assert v.numpy() == 1.0

# Reassign the value `v`
v.assign(3.0)
assert v.numpy() == 3.0

# Use `v` in a TensorFlow `tf.square()` operation and reassign
v.assign(tf.square(v))
assert v.numpy() == 9.0

class Model(object):
	def __init__(self):
		# Initialize the weights to `5.0` and the bias to `0.0`
		# In practice these should be initialized to random values
		self.W = tf.Variable(5.0)
		self.b = tf.Variable(0.0)

	def __call__(self, x):
		return self.W * x + self.b

model = Model()

assert model(3.0).numpy() == 15.0

def loss(predicted_y, target_y):
	return tf.reduce_mean(tf.square(predicted_y - target_y))

TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000

inputs = tf.random.normal(shape=[NUM_EXAMPLES])
noise = tf.random.normal(shape=[NUM_EXAMPLES])
outputs = inputs * TRUE_W + TRUE_b + noise

import matplotlib.pyplot as plt

plt.scatter(inputs, outputs, c='b')
plt.scatter(inputs, model(inputs), c='r')
plt.show()

print('Current loss: %1.6f' % loss(model(inputs), outputs).numpy())

def train(model, inputs, outputs, learning_rate):
	with tf.GradientTape() as t:
		current_loss = loss(model(inputs), outputs)
	dW, db = t.gradient(current_loss, [model.W, model.b])
	model.W.assign_sub(learning_rate * dW)
	model.b.assign_sub(learning_rate * db)

model = Model()

# Collect the history of W-values and b-values to plot later
Ws, bs = [], []
epochs = range(10)
for epoch in epochs:
	Ws.append(model.W.numpy())
	bs.append(model.b.numpy())
	current_loss = loss(model(inputs), outputs)

	train(model, inputs, outputs, learning_rate=0.1)
	print("Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f" %
		(epoch, Ws[-1], bs[-1], current_loss))

# Let's plot it all
plt.plot(epochs, Ws, 'r',
	epochs, bs, 'b')
plt.plot([TRUE_W] * len(epochs), 'r--',
		[TRUE_b] * len(epochs), 'b--')
plt.legend(['W', 'b', 'True W', 'True b'])
plt.show()



