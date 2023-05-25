import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from tabulate import tabulate
from openpyxl.styles import PatternFill
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl import Workbook
import itertools
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D
import  matplotlib
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
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pydot

from loto_function import *

#%% upload files
path = "./data/"
loto_type = "euromillions" # "loto" or "euromillions"
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

#%% derniere ligne numéro

# np.savetxt(path+'derniere_boule_'+loto_type+'.csv', derniere_ligne(boules,loto_type), delimiter=",", fmt='%d', comments='')
# np.savetxt(path+'derniere_etoiles_'+loto_type+'.csv', derniere_ligne(etoiles,loto_type), delimiter=",", fmt='%d', comments='')


#%% stats

# hist_par_colonnes(boules)
# hist_par_colonnes(etoiles)

# hist_(boules)
# hist_(etoiles)

# stat_m(boules)
# stat_m(etoiles)

# trouver_lignes_identiques(boules)

# kl_distrib(boules)

# verif_comb_etoiles(etoiles)
# verif_comb_duo_boules(boules,loto_type)


#%% deep learning
# 1. Load the data
# 2. Define the model architecture
# 3. Compile the model
# 4. Train the model

# Load the data
series_size = 100
num_samples = tableau_boules.shape[0] - series_size
x_train, y_train = create_data(tableau_boules,series_size)

#%% find sequence
# sequece = [1,3,5,7,9]
# find_row_boules(boules,sequece)
#
# print('\n binomes')
# seq = generer_binomes(sequece)
# for seq_ in seq:
#     find_row_boules(boules,seq_)
#
# print('\n trinomes')
# #
# seq = generer_trinomes(sequece)
# for seq_ in seq:
#     find_row_boules(boules,seq_)
#
# print('\n quadrinomes')
# #
# seq = generer_quadrinomes(sequece)
# for seq_ in seq:
#     find_row_boules(boules,seq_)

#%% stat to excel
#
tab_tot, tab_par_boules = frequence_calc (boules,loto_type)
tab_tot_etoiles, tab_par_etoiles = frequence_calc (etoiles,loto_type)

# tab_tot, tab_par_boules = frequence_calc (boules,loto_type,trier_=False)
# tab_tot_etoiles, tab_par_etoiles = frequence_calc (etoiles,loto_type, trier_=False)

#%%  Define the model architecture
from tensorflow.keras.callbacks import Callback
from tqdm import tqdm

# #%%  Define the model architecture
# # Définir les dimensions des tenseurs d'entrée et de sortie
# input_shape = (7, series_size)
# output_shape = (1, 7)
#
#
# # première version
# # Entrée du réseau
# # inputs = Input(shape=input_shape)
# #
# # # Liste pour stocker les sorties de chaque réseau parallèle
# # outputs = []
# #
# # # Diviser l'entrée en une liste de tenseurs
# # input_list = tf.split(inputs, num_or_size_splits=input_shape[1], axis=1)
# #
# # # Créer un réseau parallèle pour chaque partie de l'entrée
# # for i in range(input_shape[1]):
# #     # Sélectionner la ième partie de l'entrée
# #     partition = input_list[i]
# #     # Couche cachée pour chaque réseau parallèle
# #     hidden = Dense(64, activation='relu')(partition)
# #     # Couche de sortie pour chaque réseau parallèle
# #     out = Dense(output_shape[1], activation='linear')(hidden)
# #     # Remodeler la sortie dans la dimension souhaitée
# #     reshaped_out = Reshape((1, output_shape[1]))(out)
# #     # Ajouter la sortie à la liste des sorties
# #     outputs.append(reshaped_out)
# #
# # # Concaténer les sorties des réseaux parallèles
# # concatenated = tf.keras.layers.Concatenate(axis=1)(outputs)
# #
# # # Créer le modèle
# # model = Model(inputs=inputs, outputs=concatenated)
#
#
# # deuxième version
# # # Entrée du réseau
# inputs = Input(shape=input_shape)
#
# # flatten = Flatten()(inputs)
# # encoded = Dense(1024, activation='relu')(flatten)
#
# encoded = LSTM(1024, return_sequences=False)(inputs)
# decoded = Dense(7)(encoded)
# output = Reshape((1, 7))(decoded)
# model = Model(inputs, output)
#
#
# model.summary()
#
# # Compile the model
# optimizer = Adam(learning_rate=0.001)
# model.compile(optimizer=optimizer, loss='mse')
#
# # Define the callbacks
# num_epochs = 33000
# progress_callback = ProgressCallback(num_epochs)
# early_stopping = EarlyStopping(monitor='loss', patience=1000, restore_best_weights=True)
# reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.95, patience=100, min_lr=0.000001)
# save_model = SaveModelOnThreshold(filepath='model_checkpoint.h5', monitor='loss', threshold=0.001)
#
# # Train the model
# model.fit(x_train, y_train, batch_size=1024, epochs=num_epochs, verbose=0, callbacks=[progress_callback, early_stopping, reduce_lr, save_model])
#
#
#
#
# #%% test
# # Sélectionner un échantillon d'entrée pour la prédiction
# x_test = x_train[-1]
# x_test = np.expand_dims(x_test, axis=0)
#
# # Effectuer la prédiction
# prediction = model.predict(x_test)
# #
# # # Afficher la prédiction
# print(prediction.astype(int))
# #
# tab = np.transpose(tableau_boules)
# input = tf.convert_to_tensor(tab[:, 0+num_samples:series_size+num_samples].astype(np.int32))
# input = np.expand_dims(input, axis=0)
# prediction = model.predict(input)
# print(prediction.astype(int))