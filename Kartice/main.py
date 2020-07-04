from pip._internal.utils.misc import tabulate
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import operator
from sklearn import metrics
sns.set()


#radi potpunog prikaza prilikom print naredbe
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
"""

                    !!!CILJ JE GRUPISATI KORISNIKE SLICNOG PONASANJA!!!

"""
podaci = pd.read_csv('./credit_card_data.csv')

def deskriptivnaStatistika():
    #prikazuje zaglavlja, koje kolone imamo
    print(podaci.head())
    print("\n\n\n")
    #daje nam info o broju vrsta i null vrijednosti za svaku kolonu
    podaci.info()
    """Zakljucujemo da samo minimum payments ima veci broj null vrijednosti,
       ali to je mali broj null vrijednosti koji nam ne smeta.
       CUST_ID smatram da se moze zanemariti, jer nam nije od znacaja. U nastavku koda ce biti izbacen.
       Isto vazi i za TENURE.
    """
    del podaci['CUST_ID']
    del podaci['TENURE']
    print("\n\n\n")
    # deskriptivna statistika za sve kolone
    print(podaci.describe())

def raspodjelaSvihKolona():
    # raspodjela svih kolona
    fig, axes = plt.subplots(nrows=3, ncols=3)
    im = 0
    for i, column in enumerate(podaci.columns):
        sns.distplot(podaci[column], ax=axes[i // 3, i % 3])
        im += 1
        if im == 9:
            break

    fig, axes = plt.subplots(nrows=3, ncols=3)
    im = 0
    for i, column in enumerate(podaci.columns):
        im += 1
        if im > 9:
            sns.distplot(podaci[column], ax=axes[(i - 9) // 3, (i - 9) % 3])
    plt.show()


def histogramRaspodjeleSvihKolona():
    # histogram raspodjele podataka svih kolona
    podaci.hist(figsize=(20, 20), bins=50, xlabelsize=6, ylabelsize=3)
    plt.show()

def heatMapZaSveKolone():
    # heatmap koja pokazuje korelaciju izmedju podataka
    """Podaci koji nisu u korelaciji jedni sa drugima, potrebno ih je izbaciti. Ako je vise podataka
    u korelaciji sa nekim podatkom A, onda moramo provjeriti koji od tih podataka koji su u korelaciji sa
    podatkom A su u korelaciji jedni sa drugim. Oni koji jesu, jedan od njih zadrzavamo a ostale izbacimo"""
    plt.figure()
    sns.heatmap(podaci.corr(), annot=True)
    plt.show()


def boxplotSvihKolona():
    # boxplot svih kolona
    for i, column in enumerate(podaci.columns):
        plt.subplots()
        sns.boxplot(podaci[column],orient='v')
    plt.show()

def fillNaN():
    """Popuni null vrijednosti srednjom vrijednoscu te kolone"""
    mean = np.mean(podaci['MINIMUM_PAYMENTS'])
    podaci['MINIMUM_PAYMENTS'].fillna(mean,inplace=True)
    mean = np.mean(podaci['CREDIT_LIMIT'])
    podaci['CREDIT_LIMIT'].fillna(mean, inplace=True)

def scale():
    """OBAVEZNO SKALIRATI
        NA NETU PISE ZA KMeans DA SE MORA OBAVEZNO SKALIRATI
        JER SE KORISTI DISTANCA KOJA JE OSJETLJIVA NA OVO"""

    scaler = StandardScaler()
    return scaler.fit_transform(podaci)

def resavanjeOutleyera():
    """Rjesavanje outleyera tako sto se primijeni logaritam nad svakom celijom
    +1 zbog vrijednosti koje su 0"""
    return podaciBezLayeraSkalirani.applymap(lambda x: np.log(x + 1))

def izbacivanjeSuvisnogBrojaAtributa():
    """Smanjuje dimenzionalnost podataka"""
    pc = PCA(n_components=5)
    pcfit=pc.fit(podaciBezLayeraSkalirani)
    smanjeniAtr = pcfit.fit_transform(podaciBezLayeraSkalirani)
    return smanjeniAtr

def odredjivanjePotrebnogBrojaAtributa():
    """Odredjuje se potreban broj atributa da se smanji dimenzionalnost
    podataka radi lakseg klasterovanja"""
    brojAtr = []
    suma = []
    for i in range(2, 14):
        pc = PCA(n_components=i)
        pcfit = pc.fit(podaciBezLayeraSkalirani)
        suma=0
        for v in pcfit.explained_variance_ratio_:
            suma+=v
        suma.append(suma)
        brojAtr.append(i)
    plt.plot(brojAtr,suma, marker='x')
    plt.show()

if __name__ == '__main__':
    deskriptivnaStatistika()
    fillNaN()
    histogramRaspodjeleSvihKolona()
    boxplotSvihKolona()
    heatMapZaSveKolone()
    podaciBezLayeraSkalirani = resavanjeOutleyera()
    kolone=podaciBezLayeraSkalirani.columns
    podaciBezLayeraSkalirani = scale()
    odredjivanjePotrebnogBrojaAtributa()
    skraceniPodaci = izbacivanjeSuvisnogBrojaAtributa()
