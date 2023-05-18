import requests
from bs4 import BeautifulSoup
import zipfile
import io
import pandas as pd
import chardet
import numpy as np

#%%
def lecture_fichier_csv(fichier_zip, nom_fichier, loto='loto'):
    fichier_csv = fichier_zip.open(nom_fichier)

    # Utiliser chardet pour détecter l'encodage du fichier
    encodage_detecte = chardet.detect(fichier_csv.read())

    # Rembobiner le curseur du fichier pour le relire
    fichier_csv.seek(0)

    # Lire le contenu du fichier avec l'encodage détecté
    contenu_csv = fichier_csv.read().decode(encodage_detecte['encoding'])

    # Lire le fichier CSV avec Pandas
    df = pd.read_csv(io.StringIO(contenu_csv), sep=';')


    # recuperer les colonnes qui nous interessent : date et boules
    if loto == 'loto':
        df = df.iloc[:, 4:10]
        # permuter l'ordre des lignes decroissantes
        df = df[::-1]
        df = df.to_numpy()

    elif loto == 'euromillions':
        if nom_fichier.startswith('euromillions_4'):
            df = df.iloc[:,4:11]
            df = df.to_numpy()
        elif nom_fichier.startswith('euromillions_3'):
            df = df.iloc[:, 4:11]
            df = df.to_numpy()
        else: # recupere les collones 5 à 11 seulement
            df = df.iloc[:, 5:12]
            df = df[::-1]
            df = df.to_numpy()


    fichier_csv.close()

    return df


def recuperer_numeros_loto(loto='loto'):
    # Récupérer la page html
    if loto == 'loto':
        url = 'https://www.fdj.fr/jeux-de-tirage/loto/historique'
    elif loto == 'euromillions':
        url = 'https://www.fdj.fr/jeux-de-tirage/euromillions-my-million/historique'
    else:
        print('Le jeu de loto demandé n\'existe pas')
        return

    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')

    # Trouver les liens vers les fichiers zip
    liens_zip = soup.find_all('a', href=lambda href: href and href.endswith('.zip'))

    fichier_zip_list = []
    boules = []
    for lien_zip in liens_zip:
        lien = lien_zip['href']
        nom_fichier_zip = lien.split('/')[-1]
        # si le nom du fichier zip commence par loto et se termine par une année supérieure à 2000
        t  = float(nom_fichier_zip[-8:-4])
        if loto == 'loto':
            if nom_fichier_zip.startswith('loto_') and float(nom_fichier_zip[-10:-6]) > 2000:
                # print("Lien vers le fichier zip : ", lien)
                # Télécharger le fichier zip
                response = requests.get(lien)
                fichier_zip = zipfile.ZipFile(io.BytesIO(response.content))
                fichier_zip_list.append(fichier_zip)


        elif loto == 'euromillions':
            if nom_fichier_zip.startswith('euromillions_') and float(nom_fichier_zip[-10:-6]) >= 2014:
                # print("Lien vers le fichier zip : ", lien)
                # Télécharger le fichier zip
                response = requests.get(lien)
                fichier_zip = zipfile.ZipFile(io.BytesIO(response.content))
                fichier_zip_list.append(fichier_zip)


    # Extraire le fichier csv contenu dans le zip , filtré uniquement les fichier loto, et le lire avec la librairie csv
    for fichier_zip in fichier_zip_list:
        for nom_fichier in fichier_zip.namelist():
            if nom_fichier.endswith('.csv'):
                fichier_csv = fichier_zip.open(nom_fichier)

                # Extraire le fichier csv contenu dans le zip
                for nom_fichier in fichier_zip.namelist():
                    if nom_fichier.endswith('.csv'):
                        # insertion des boules dans la liste au dessus
                        boules.insert(0, lecture_fichier_csv(fichier_zip, nom_fichier, loto))


                        print("Fichier csv : ", nom_fichier, " ", boules[0].shape)


    boules = np.concatenate(boules, axis=0)
    print(boules.shape)
    # sauvegarder les boules dans un fichier csv : tableau d'entier uniquement
    np.savetxt('./data/boules_'+loto+'.csv', boules, delimiter=',', fmt='%d')




loto = ['euromillions'] # ['loto', 'euromillions']
recuperer_numeros_loto(loto[0])

loto = ['loto'] # ['loto', 'euromillions']
recuperer_numeros_loto(loto[0])

#%%