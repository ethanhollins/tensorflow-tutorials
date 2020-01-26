import tensorflow as tf
from tensorflow import keras

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

import numpy as np

(train_data, test_data), info = tfds.load(
	'imdb_reviews/subwords8k',
	split = (tfds.Split.TRAIN, tfds.Split.TEST),
	as_supervised=True,
	with_info=True
)

encoder = info.features['text'].encoder

print('Vocabulary size: {}'.format(encoder.vocab_size))

sample_string = 'Hello Tensorflow'

encoded_string = encoder.encode(sample_string)
print('Encoded string is {}'.format(encoded_string))

original_string = encoder.decode(encoded_string)
print('The original string: {}'.format(original_string))

assert original_string == sample_string

for ts in encoded_string:
	print('{} ----> {}'.format(ts, encoder.decode([ts])))

for train_example, train_label in train_data.take(1):
	print('Encoded text:', train_example[:10].numpy())
	print('Label:', train_label.numpy())

	print(encoder.decode(train_example))

BUFFER_SIZE = 1000

train_batches = (
	train_data
	.shuffle(BUFFER_SIZE)
	.padded_batch(32, train_data.output_shapes)
)

test_batches = (
	test_data
	.padded_batch(32, train_data.output_shapes)
)

for example_batch, label_batch in train_batches.take(2):
	print("Batch shape:", example_batch.shape)
	print("label shape:", label_batch.shape)

model = keras.Sequential([
	keras.layers.Embedding(encoder.vocab_size, 16),
	keras.layers.GlobalAveragePooling1D(),
	keras.layers.Dense(1, activation='sigmoid')
])

print(model.summary())

model.compile(
	optimizer='adam',
	loss='binary_crossentropy',
	metrics=['accuracy']
)

history = model.fit(
	train_batches,
	epochs=10,
	validation_data=test_batches,
	validation_steps=30
)

loss, accuracy = model.evaluate(test_batches)
print("Loss: ", loss)
print("Accuracy: ", accuracy)

history_dict = history.history
print(history_dict.keys())

import matplotlib.pyplot as plt

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


plt.clf()
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()