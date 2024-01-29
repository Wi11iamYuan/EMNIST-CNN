#%%
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras import layers 

from emnist import list_datasets
from emnist import extract_training_samples
from emnist import extract_test_samples

#%%
# getting the data
images_train, labels_train = extract_training_samples('letters')
images_test, labels_test = extract_test_samples('letters')
#%%
plt.imshow(images_train[0])
print(labels_train[0])

plt.imshow(images_test[0])
print(labels_test[0])
# %%
#normalizing between 0 and 1
images_train, images_test = images_train / 255.0 , images_test / 255.0
# %%
#convolutional neural network, sequential, step by step
model = keras.models.Sequential()
#convultion + pooling
model.add(layers.Conv2D(28, (3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
# same as above but instead utilizing default values
model.add(layers.Conv2D(28, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

#flatten
model.add(layers.Flatten())
#dense layer
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(27))

print(model.summary())
# %%
#soft max and cross entropy go hand in hand (sparse for if labels are integers)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#gradient descent, learning rate is how fast it goes down the gradient
optim = keras.optimizers.Adam(learning_rate=0.001)
metrics = ['accuracy']
# %%
model.compile(optimizer=optim, loss=loss, metrics=metrics)
# %%
#training 
#how many images to train at a time
batch_size = 64
#how many times to run through all the data
epochs = 5
# %%
model.fit(images_train, labels_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2)
# %%
model.evaluate(images_test, labels_test, batch_size=batch_size, verbose=2)