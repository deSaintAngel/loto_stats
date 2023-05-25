from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, TimeDistributed, RepeatVector, Flatten
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

#%% classe data_set_loto
class data_set_loto:
    def __init__(self, path, loto_type, series_size=12):
        self.path = path
        self.loto_type = loto_type
        self.tableau_boules = np.genfromtxt(path + "boules_" + loto_type + ".csv", delimiter=',', skip_header=1)
        self.boules = self.tableau_boules[:, :5]
        self.etoiles = self.tableau_boules[:, 5:]
        self.series_size = series_size
        self.num_samples = self.tableau_boules.shape[0] - self.series_size
        self.size_dim = self.tableau_boules.shape[1]

        self.train_boules, self.label_boules, self.scaler_boules = self.dataset(
            self.tableau_boules, self.series_size, self.size_dim)

        self.freq_boules = self.get_freq_boules(self.boules)
        self.freq_etoiles = self.get_freq_boules(self.etoiles)
        self.freq_ = np.concatenate((self.freq_boules, self.freq_etoiles), axis=1)
        self.train_freq_boules, self.label_freq_boules, self.scaler_freq_boules = self.dataset(
            self.freq_, self.series_size, int(np.max(self.boules))+int(np.max(self.etoiles)), scaler=False)

        self.last_boules = self.get_last_boules(self.boules)
        self.last_etoiles = self.get_last_boules(self.etoiles)
        self.last_ = np.concatenate((self.last_boules, self.last_etoiles), axis=1)
        self.train_last_boules, self.label_last_boules, self.scaler_last_boules = self.dataset(
            self.last_, self.series_size, int(np.max(self.boules))+int(np.max(self.etoiles)), scaler=True)

        self.tab_min_max = self.get_min_max(self.boules)
        self.tab_min_max_etoiles = self.get_min_max(self.etoiles)
        if self.loto_type == "euromillions":
            self.tab_min_max = np.concatenate((self.tab_min_max, self.tab_min_max_etoiles), axis=1)
        self.train_min_max, self.label_min_max, self.scaler_min_max = self.dataset(
            self.tab_min_max, self.series_size, int(self.tab_min_max.shape[-1]), scaler=True)


    def dataset(self, df, series_size, nb_label_feature, scaler=True):
        number_of_rows = df.shape[0]  # taille du dataset number_of_features
        number_of_features = df.shape[1]
        if scaler:
            scaler = StandardScaler().fit(df)
            df = scaler.transform(df)
        else:
            scaler = None

        train = np.empty([number_of_rows - series_size, series_size, number_of_features], dtype=float)

        label = np.empty([number_of_rows - series_size, nb_label_feature], dtype=float)
        for i in range(0, number_of_rows - series_size):
            train[i] = df[i:i + series_size, 0: number_of_features]
            label[i] = df[i + series_size: i + series_size + 1, 0:nb_label_feature]
        return train, label, scaler

    def get_freq_boules(self,tab_freq):
        """ retourne un tableaux qui pour chaque ligne de tirage donne la fréquence d'apparition
        des boules dans  les precedants tirages
        :param: tab
        :return: freq_boules
        """
        # tx valeur maximal de boule entier
        tx = int(np.max(tab_freq))
        freq_boules = np.zeros((tab_freq.shape[0], tx))
        for i in range(tab_freq.shape[0]):  # pour chaque tirage
            for j in range(tx):  # pour chaque boule
                if i == 0:
                    freq_boules[i, j] = 0  # Handle division by zero for the first draw
                else:
                    freq_boules[i, j] = np.sum(tab_freq[:i, :] == j + 1)

        # diviser chaque ligne par l'indice du tirage
        freq_boules = freq_boules / np.arange(1, tab_freq.shape[0] + 1).reshape(-1, 1)
        return freq_boules



    def get_last_boules(self,tab):
        """ retourne un tableaux qui pour chaque ligne de tirage donne le nombre de fois ou la boule n'a pas été tirée
        :param: tableau_boules
        :return: last_boules
        """
        last_boules = np.zeros((tab.shape[0], np.max(tab).astype(int)))

        for i in range(1,tab.shape[0]): # pour chaque tirage
            for j in range(1, np.max(tab).astype(int)+1): # pour chaque boule
                if j in tab[i]: # si la boule est dans le tirage précédent on remet à 0
                    last_boules[i, j-1] = 0
                else: # sinon on incrémente de 1 le nombre de tirage sans la boule
                    last_boules[i, j-1] = last_boules[i-1, j-1] + 1

        return last_boules

    def get_min_max(self,tab):
        """ retourne un tableaux qui pour chaque ligne de tirage la somme, le min et le max
        des boules tirées dans le tirage précédent
        :param: tableau_boules
        :return: last_boules
        """
        tab_min_max = np.zeros((tab.shape[0], 3))
        for i in range(1,tab.shape[0]): # pour chaque tirage
            tab_min_max[i, 0] = np.sum(tab[i-1]) # somme des boules
            tab_min_max[i, 1] = np.min(tab[i-1]) # min des boules
            tab_min_max[i, 2] = np.max(tab[i-1]) # max des boules

        return tab_min_max




    def save(self):
        pickle.dump(self, open(self.path + "data_set_loto_" + self.loto_type + ".pkl", "wb"))

    def load(self):
        return pickle.load(open(self.path + "data_set_loto_" + self.loto_type + ".pkl", "rb"))




#%% class main
if __name__ == "__main__":
    path = "./data/"
    loto_type = "loto" # "loto" or "euromillions"
    series_size = 24
    data_set_loto = data_set_loto(path, loto_type, series_size)


    print(data_set_loto.tableau_boules.shape)
    print(data_set_loto.train_boules.shape)
    print(data_set_loto.label_boules.shape)
    print(data_set_loto.train_freq_boules.shape)
    print(data_set_loto.label_freq_boules.shape)
    print(data_set_loto.train_last_boules.shape)
    print(data_set_loto.label_last_boules.shape)
    print(data_set_loto.train_min_max.shape)
    print(data_set_loto.label_min_max.shape)

#%% construction du model

