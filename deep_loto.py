import os
import matplotlib.pyplot as plt
import numpy as np
import itertools
import tensorflow as tf,keras
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.layers import LSTM
from tensorflow.keras.layers import Input, Flatten, Reshape, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Reshape
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Activation, Dropout, BatchNormalization
from tqdm import tqdm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, TimeDistributed, RepeatVector, Flatten
from keras.callbacks import EarlyStopping


from loto_function import *

#%% upload files
path = "./data/"
loto_type = "loto" # "loto" or "euromillions"
# Charger le fichier CSV dans une matrice
tableau_boules = np.genfromtxt(path+"boules_"+loto_type+".csv", delimiter=',', skip_header=1)
# tab_jeux_done = np.genfromtxt(path+"loto_jeux.csv", delimiter=',', skip_header=1)

boules = tableau_boules[:,:5]
etoiles = tableau_boules[:,5:]
# boules = np.apply_along_axis(np.sort, axis=1, arr=boules)
# etoiles = np.apply_along_axis(np.sort, axis=1, arr=etoiles)

print('tableau_boules shape', tableau_boules.shape)
print('boules shape', boules.shape)
print('etoiles shape', etoiles.shape)

#%% deep learning
# 1. Load the data
# 2. Define the model architecture
# 3. Compile the model
# 4. Train the model


# traitement data
def dataset(df, series_size, nb_label_feature):
    number_of_rows = df.shape[0]  # taille du dataset number_of_features
    number_of_features = df.shape[1]
    scaler = StandardScaler().fit(df)
    transformed_dataset = scaler.transform(df)
    train = np.empty([number_of_rows - series_size, series_size, number_of_features], dtype=float)

    label = np.empty([number_of_rows - series_size, nb_label_feature], dtype=float)
    for i in range(0, number_of_rows - series_size):
        train[i] = transformed_dataset[i:i + series_size, 0: number_of_features]
        label[i] = transformed_dataset[i + series_size: i + series_size + 1, 0:nb_label_feature]
    return train, label, scaler

# Load the data
series_size = 300
num_samples = tableau_boules.shape[0] - series_size

x_train, y_train, scaler = dataset(tableau_boules, series_size, np.shape(tableau_boules)[-1])

#%%  Define the model parameters
# Définir les dimensions des tenseurs d'entrée et de sortie
input_shape = (series_size, np.shape(x_train)[-1])

#%%  Define the model architecture
# premiere version
#Architecture du modèle
def define_model(input_shape,UNITS):
    #initialisation du rnn
    model = Sequential()
    #ajout de la premiere couche lstm
    model.add(LSTM(UNITS, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(UNITS, dropout=0.2, return_sequences=False))
    #ajout de la couche de sortie
    model.add(Dense(input_shape[-1]))
    return model

def model_dense(input_shape, UNITS=100):
    inputs = Input(shape=input_shape)
    x = Flatten()(inputs)
    x = Dense(UNITS, activation='relu')(x)
    decoded = Dense(6)(x)
    # output = Reshape((1, 6))(decoded)
    return Model(inputs, decoded)

UNITS = 64
number_of_features = np.shape(x_train)[-1]

model = define_model(input_shape,UNITS)
# model = model_dense(input_shape, UNITS)

#%% train the model
model.summary()

# Compile the model
optimizer = Adam()
model.compile(optimizer=optimizer, loss='mse')

# Define the callbacks
num_epochs = 10000
progress_callback = ProgressCallback(num_epochs)
early_stopping = EarlyStopping(monitor='loss', min_delta=0.00001, patience=1000)
# es = EarlyStopping(monitor='loss', mode='max', verbose=0, patience=100)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.95, patience=25)
save_model = SaveModelOnThreshold(filepath='./data/model_ckpt/model_checkpoint.h5', monitor='loss', threshold=0.001)

# Train the model
model.fit(x_train, y_train, batch_size=1024, epochs=num_epochs, verbose=0, callbacks=[progress_callback, reduce_lr])



# param : batch_size=1024, epochs=33000,
# early_stopping(patience)=1000, reducon_plateau(patience,factor,minlr)=(100,0.95,0.000001)
# optimizer = Adam(learning_rate=0.001), loss='mse', lmst(512, return_sequences=False), Dense(7)
#%% test
# Sélectionner un échantillon d'entrée pour la prédiction
x_test = x_train[-1]
x_test = np.expand_dims(x_test, axis=0)

# Effectuer la prédiction
prediction = model.predict(x_test)
prediction = scaler.inverse_transform(prediction)
#
# # Afficher la prédiction
print(prediction.astype(int))
#
# tab = np.transpose(tableau_boules)
input = tf.convert_to_tensor(tableau_boules[0+num_samples:series_size+num_samples,:].astype(np.int32))
input = np.expand_dims(input, axis=0)
prediction = model.predict(input)
prediction = scaler.inverse_transform(prediction)
print(prediction.astype(int))