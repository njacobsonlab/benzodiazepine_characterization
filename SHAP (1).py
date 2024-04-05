# ## installs ##
# !pip install pyts
# # Packages
# !pip install pandas
# !pip install numpy
# !pip install tensorflow
# !pip install matplotlib
# !pip install random
# !pip install os
# !pip install tensorflow
# !pip install shap
# !pip3 install warnings
# !pip3 install shap
# !pip install pyarrow

import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
import os

# Pyarrow
import pyarrow

# SHAP
import shap

# Sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, auc, roc_auc_score, f1_score, balanced_accuracy_score, confusion_matrix, recall_score, roc_curve, precision_score

# Pyts
from pyts.image import GramianAngularField

# Keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.layers import Embedding
from keras.metrics import AUC

# Tf
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.models import Sequential
import random

# Import Layers
from keras.layers import ConvLSTM2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.convolutional import MaxPooling3D
import time
start = time.time()

# Read in data
Wide_X = pd.read_parquet('Data/[Smooth]SEQN_WideX.parquet')
SEQN = Wide_X['SEQN']
X = Wide_X.drop(columns=['SEQN'])
data_test = pd.DataFrame(X) 
# No need to change -- model 6 is the best performing model 
model = keras.models.load_model('Models/full_model_6.h5')
# Dimensions
n_steps, n_length, n_width, n_features = 7, 24, 60, 1

# Scale Data
scaler = StandardScaler()
scaler.fit(data_test)
df_norm = scaler.transform(data_test)

# Reshape 
df_reshape = df_norm.reshape(df_norm.shape[0], n_steps, n_length, n_width, n_features)
background = df_reshape[np.random.choice(df_reshape.shape[0], 100, replace=False)]

#
print("Checkpoint")

# Run shap
explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(df_reshape, check_additivity = False)
shap_values = np.array(shap_values)

print("FINISHED")
np.save('results/[complete]new_shap_values_.npy', shap_values)

end = time.time()
print(end - start)
