#%%
from emnist import list_datasets
from emnist import extract_training_samples
from emnist import extract_test_samples

import matplotlib.pyplot as plt

#%%

images, labels = extract_training_samples('letters')
#%%
plt.imshow(images[0])
print(labels[0])
# %%
