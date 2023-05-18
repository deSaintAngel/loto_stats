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
import numpy as np

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
boules = np.apply_along_axis(np.sort, axis=1, arr=boules)
etoiles = np.apply_along_axis(np.sort, axis=1, arr=etoiles)

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
# series_size = 100
# num_samples = tableau_boules.shape[0] - series_size
# x_train, y_train = create_data(tableau_boules,series_size)

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
# tab_tot, tab_par_boules = frequence_calc (boules,loto_type)
# tab_tot_etoiles, tab_par_etoiles = frequence_calc (etoiles,loto_type)
#
# tab_tot, tab_par_boules = frequence_calc (boules,loto_type,trier_=False)
# tab_tot_etoiles, tab_par_etoiles = frequence_calc (etoiles,loto_type, trier_=False)


#%% stat suite de 2 nombres consécutifs
suite_binome_ = suite_trinome(boules)
print(suite_binome_)