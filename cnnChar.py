#%%
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras import layers 
import numpy as np

from emnist import list_datasets
from emnist import extract_training_samples
from emnist import extract_test_samples

#%%
images_train, labels_train = extract_training_samples('letters')
images_train = list(images_train)
labels_train = list(labels_train)

images_temp, labels_temp = extract_training_samples('digits')
images_train += list(images_temp)
labels_train += list(labels_temp + 27)

# images_train = tuple(images_train)
# labels_train = tuple(labels_train)
#%%
images_test, labels_test = extract_test_samples('letters')
images_test = list(images_test)
labels_test = list(labels_test)

images_temp, labels_temp = extract_test_samples('digits')
images_test += list(images_temp)
labels_test += list(labels_temp + 27)

# images_test = tuple(images_test)
# labels_test = tuple(labels_test)

#%%
plt.imshow(images_test[-1])
print(labels_test[-1])
#%%
images_train = np.array(images_train)
labels_train = np.array(labels_train)
images_test = np.array(images_test)
labels_test = np.array(labels_test)

images_train, images_test = images_train / 255.0 , images_test / 255.0
#%%
model = keras.models.Sequential()
model.add(layers.Conv2D(28, (3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(28, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(37))
#%%
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(learning_rate=0.001)
metrics = ['accuracy']
#%%
model.compile(optimizer=optim, loss=loss, metrics=metrics)
#%%
batch_size = 64
epochs = 5
#%%
model.fit(images_train, labels_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2)
# %%
model.evaluate(images_test, labels_test, batch_size=batch_size, verbose=2)