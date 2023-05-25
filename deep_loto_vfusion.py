from tqdm import tqdm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler

from keras.callbacks import Callback

from tensorflow.keras.layers import concatenate

from loto_function import *
from data_deep_preload import data_set_loto
import sys

#%%
from tqdm import tqdm
class TqdmCallback(Callback):
    def __init__(self, total_epochs):
        super(TqdmCallback, self).__init__()
        self.total_epochs = total_epochs
        self.pbar = None
        self.loss_history = []

    def on_train_begin(self, logs=None):
        self.pbar = tqdm(total=self.total_epochs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.loss_history.append(logs.get('loss'))
        self.loss_history = self.loss_history[-5:]  # Garder les cinq dernières valeurs de loss
        self.pbar.update(1)
        self.pbar.set_description(f"Epoch {epoch + 1}/{self.total_epochs}")
        self.pbar.set_postfix(loss=', '.join([f'{loss:.4f}' for loss in self.loss_history]))

    def on_train_end(self, logs=None):
        self.pbar.close()

#%% upload files
path = ""
loto_type = "loto" # "loto" or "euromillions"
# Charger le fichier CSV dans une matrice

#%%  Load the data
series_size = 36

data_set_loto = data_set_loto(path, loto_type, series_size)
# sauvegarde de data_set_loto
with open(f'data_set_{loto_type}.pkl', 'wb') as f:
    pickle.dump(data_set_loto, f)

with open(f'data_set_{loto_type}.pkl', 'rb') as f:
    data_set_loto = pickle.load(f)
#
# print(data_set_loto.tableau_boules.shape)
#
# print(data_set_loto.train_boules.shape)
# print(data_set_loto.label_boules.shape)
#
# print(data_set_loto.train_freq_boules.shape)
# print(data_set_loto.label_freq_boules.shape)
#
# print(data_set_loto.train_last_boules.shape)
# print(data_set_loto.label_last_boules.shape)
#
# print(data_set_loto.train_min_max.shape)
# print(data_set_loto.label_min_max.shape)

#%%  build the model
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate
from tensorflow.keras.models import Model

def build_model(series, loto_type, nbr_neurone_h1, nbr_neurone_h2):
    if loto_type == 'loto':
        l = 3
        n_max = 59
        n = 6
    elif loto_type == 'euromilion':
        l = 6
        n_max = 62
        n = 7

    # Première branche : model_boules
    input_boules = Input(shape=(series, n))
    lstm_boules_h1 = LSTM(nbr_neurone_h1, return_sequences=True)(input_boules)
    lstm_boules_h2 = LSTM(nbr_neurone_h2, return_sequences=False)(lstm_boules_h1)
    output_boules = Dense(n)(lstm_boules_h2)
    model_boules = Model(inputs=input_boules, outputs=output_boules)
    model_boules.compile(loss='mse', optimizer='adam')

    # Deuxième branche : model_freq
    input_freq = Input(shape=(series, n_max))
    lstm_freq_h1 = LSTM(nbr_neurone_h1, return_sequences=True)(input_freq)
    lstm_freq_h2 = LSTM(nbr_neurone_h2, return_sequences=False)(lstm_freq_h1)
    output_freq = Dense(n_max)(lstm_freq_h2)
    model_freq = Model(inputs=input_freq, outputs=output_freq)
    model_freq.compile(loss='mse', optimizer='adam')

    # Troisième branche : model_last
    input_last = Input(shape=(series, n_max))
    lstm_last_h1 = LSTM(nbr_neurone_h1, return_sequences=True)(input_last)
    lstm_last_h2 = LSTM(nbr_neurone_h2, return_sequences=False)(lstm_last_h1)
    output_last = Dense(n_max)(lstm_last_h2)
    model_last = Model(inputs=input_last, outputs=output_last)
    model_last.compile(loss='mse', optimizer='adam')

    # Quatrième branche : model_stats
    input_stats = Input(shape=(series, l))
    lstm_stats_h1 = LSTM(nbr_neurone_h1, return_sequences=True)(input_stats)
    lstm_stats_h2 = LSTM(nbr_neurone_h2, return_sequences=False)(lstm_stats_h1)
    output_stats = Dense(l)(lstm_stats_h2)
    model_stats = Model(inputs=input_stats, outputs=output_stats)
    model_stats.compile(loss='mse', optimizer='adam')

    # Cinquième branche : modèle final
    # Concaténer les entrées de chaque branche
    input_ = concatenate([input_boules, input_freq, input_last, input_stats])
    lstm_concat_h1 = LSTM(nbr_neurone_h1, return_sequences=True)(input_)
    lstm_concat_h2 = LSTM(nbr_neurone_h2, return_sequences=False)(lstm_concat_h1)

    # Relier les branches précédentes au modèle final en récupérant les prédictions de chaque branche
    pred_boules = model_boules(input_boules)
    pred_freq = model_freq(input_freq)
    pred_last = model_last(input_last)
    pred_stats = model_stats(input_stats)

    # Concaténer les sorties des branches précédentes avec lstm_concat_h2
    merged_output = Concatenate()([lstm_concat_h2, pred_boules, pred_freq, pred_last, pred_stats])

    # Sortie finale
    output_final = Dense(n)(merged_output)

    model_final = Model(inputs=[input_boules, input_freq, input_last, input_stats],
                        outputs=[output_final, output_freq, output_last, output_stats])
    model_final.compile(loss='mse', optimizer='adam')

    return model_final

# Paramètres du modèle
series = series_size
loto_type = 'loto'
nbr_neurone_h1 = 64
nbr_neurone_h2 = 32

# Construction du modèle
model = build_model(series, loto_type, nbr_neurone_h1, nbr_neurone_h2)

# Afficher un résumé du modèle
model.summary()



train_boules = data_set_loto.train_boules
label_boules = data_set_loto.label_boules
scaler_boules = data_set_loto.scaler_boules

train_freq_boules = data_set_loto.train_freq_boules
label_freq_boules = data_set_loto.label_freq_boules


train_last_boules = data_set_loto.train_last_boules
label_last_boules = data_set_loto.label_last_boules

train_min_max = data_set_loto.train_min_max
label_min_max = data_set_loto.label_min_max

#%%  train the model
EarlyStopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100)
ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=0.0000000001)

model.fit(
    [train_boules, train_freq_boules, train_last_boules, train_min_max],
    [label_boules, label_freq_boules, label_last_boules, label_min_max],
    epochs=1000,
    callbacks=[EarlyStopping, ReduceLROnPlateau],
    verbose=1
)


#%%  save the model
model.save('./data/model.h5')

#%%  load the model
from tensorflow.keras.models import load_model
model = load_model('./data/model.h5')

#%%  test the model
# generer une donnée d'entree pour tester le model la concatenation des 4 entrees du model final
# recuperer les derniers series-tirages de chaque entree
input_boules = train_boules[-1:]
input_freq = train_freq_boules[-1:]
input_last = train_last_boules[-1:]
input_stats = train_min_max[-1:]
input = [input_boules, input_freq, input_last, input_stats]
# predire la sortie
pred = model.predict(input)
prediction = scaler_boules.inverse_transform(pred[0])
y = scaler_boules.inverse_transform(label_boules[-1:])
#
# # Afficher la prédiction
print(f'prediction: {prediction.astype(int)}')
print(f'vrai tirage: {y}')

#%%  test the model
input_boules_prop = label_boules[-series:]
input_boules_prop = np.expand_dims(input_boules_prop, axis=0)
input_freq_prop = label_freq_boules[-series:]
input_freq_prop = np.expand_dims(input_freq_prop, axis=0)
input_last_prop = label_last_boules[-series:]
input_last_prop = np.expand_dims(input_last_prop, axis=0)
input_stats_prop = label_min_max[-series:]
input_stats_prop = np.expand_dims(input_stats_prop, axis=0)
input_prop = [input_boules_prop, input_freq_prop, input_last_prop, input_stats_prop]
# predire la sortie
pred_prop = model.predict(input_prop)
prediction_prop = scaler_boules.inverse_transform(pred_prop[0])
print(f'prediction: {prediction_prop.astype(int)}')



