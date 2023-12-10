#%%
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras import layers 

from emnist import list_datasets
from emnist import extract_training_samples
from emnist import extract_test_samples

#%%

images_train, labels_train = extract_training_samples('letters')
images_test, labels_test = extract_test_samples('letters')
#%%
plt.imshow(images_train[0])
print(labels_train[0])

plt.imshow(images_test[0])
print(labels_test[0])
# %%
images_train, images_test = images_train / 255.0 , images_test / 255.0
# %%

# %%
