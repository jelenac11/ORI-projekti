from IPython.display import display, HTML
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, _tree
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

"""def raspodjelaSvihKolona():
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
"""

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
    sns.heatmap(podaciBezLayeraSkalirani.corr(), annot=True)
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
    return scaler.fit_transform(podaciBezLayeraSkalirani)

def odredjivanjeBrojaKlastera():
    """Odredjuje se potreban broj klastera"""
    greske = []
    brojKlastera = []
    for brKlastera in range(1,15):
        rez = KMeans(brKlastera)
        rez.fit(skraceniPodaci)
        greske.append(rez.inertia_)
        brojKlastera.append(brKlastera)

    plt.plot(brojKlastera, greske, marker="x")
    plt.show()


def odredjivanjePotrebnogBrojaAtributa():
    """Odredjuje se potreban broj atributa da se smanji dimenzionalnost
    podataka radi lakseg klasterovanja"""
    brojAtr = []
    suma = []
    for i in range(2, 14):
        pc = PCA(n_components=i)
        pcfit = pc.fit(podaciBezLayeraSkalirani)
        s=0
        for v in pcfit.explained_variance_ratio_:
            s+=v
        suma.append(s)
        brojAtr.append(i)
    plt.plot(brojAtr,suma, marker='x')
    plt.show()

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

def kreirajiPrikaziKlastere():
    km = KMeans(n_clusters=4, random_state=1)
    klasteri = km.fit_predict(skraceniPodaci)
    print(pd.Series(km.labels_).value_counts())
    """for i in range(0,5):
        for j in range(0, 5):
            plt.subplots()
            plt.scatter(skraceniPodaci[:, i], skraceniPodaci[:, j], c=km.labels_, cmap='Spectral', alpha=0.5)
            plt.xlabel(i)
            plt.ylabel(j)
    plt.show()"""
    cluster_report(orgPodaci,klasteri)

    fig, ax = plt.subplots()
    scatter = ax.scatter(skraceniPodaci[:, 1], skraceniPodaci[:, 3], c=km.labels_, cmap='Spectral', alpha=0.5)

    plt.xlabel(1)
    plt.ylabel(3)
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="lower left", title="Klasteri")
    ax.add_artist(legend1)
    plt.show()

    #bar plot broja korisnika u svakom klasteru
    broj = pd.Series(km.labels_).value_counts()
    plt.bar(broj.index.values,broj.values)
    plt.show()

    #dodjela broja klastera svakom korisniku
    podaciSaKlasterom = pd.concat([orgPodaci, pd.Series(km.labels_, name='Grupa')], axis=1)
    #racunanje srednje vrijednosti svake kolone za svaki klaster
    print(podaciSaKlasterom.groupby("Grupa").mean().T)
    podaciSaKlasterom = podaciSaKlasterom.applymap(lambda x: np.log(x + 1))
    podaciSaKlasterom = podaciSaKlasterom.groupby("Grupa").mean()

    # bar plot prikaz svakog klastera
    plt.subplots()
    podaciSaKlasterom.iloc[0,:].plot(kind='bar')

    plt.subplots()
    podaciSaKlasterom.iloc[1,:].plot(kind='bar')

    plt.subplots()
    podaciSaKlasterom.iloc[2,:].plot(kind='bar')

    plt.subplots()
    podaciSaKlasterom.iloc[3,:].plot(kind='bar')

    plt.show()
def dodajNoveKolone():
    #dodavanje novih kolona
    podaci['AVG_PURCHASE'] = (podaci['PURCHASES'] + 1) / (podaci['PURCHASES_TRX'] + 1)
    podaci['AVG_CASH_ADVANCE'] = (podaci['CASH_ADVANCE'] + 1) / (podaci['CASH_ADVANCE_TRX'] + 1)


def odrediTipKupovine(value):
    #odredjuje se tip kupovine korisnika
    if value['ONEOFF_PURCHASES_FREQUENCY'] == 0 and value['PURCHASES_INSTALLMENTS_FREQUENCY'] == 0:
        return 'NOT_BUYING'
    elif value['ONEOFF_PURCHASES_FREQUENCY'] > 0 and value['PURCHASES_INSTALLMENTS_FREQUENCY'] == 0:
        return 'ONEOFF'
    elif value['ONEOFF_PURCHASES_FREQUENCY'] == 0 and value['PURCHASES_INSTALLMENTS_FREQUENCY'] > 0:
        return 'INSTALLMENTS'
    elif value['ONEOFF_PURCHASES_FREQUENCY'] > 0 and value['PURCHASES_INSTALLMENTS_FREQUENCY'] > 0:
        return 'BOTH'

def izbaciAtribute():
    #izbacuju se suvisni atributi
    podaciBezLayeraSkalirani.drop(['TYPE_OF_BUYING','BALANCE','PURCHASES','ONEOFF_PURCHASES','INSTALLMENTS_PURCHASES','CASH_ADVANCE','PAYMENTS','MINIMUM_PAYMENTS','PRC_FULL_PAYMENT','CREDIT_LIMIT'],axis=1,inplace=True)
    print(podaciBezLayeraSkalirani.head())

def dodajDummy():
    #dodaju se dummys na osnovu tipa kupovine
    dummy = pd.concat([podaci, pd.get_dummies(podaci['TYPE_OF_BUYING'])], axis=1)
    return dummy

def get_class_rules(tree: DecisionTreeClassifier, feature_names: list):
    inner_tree: _tree.Tree = tree.tree_
    classes = tree.classes_
    class_rules_dict = dict()


    def tree_dfs(node_id=0, current_rule=[]):
        # feature[i] holds the feature to split on, for the internal node i.
        split_feature = inner_tree.feature[node_id]
        if split_feature != _tree.TREE_UNDEFINED:  # internal node
            name = feature_names[split_feature]
            threshold = inner_tree.threshold[node_id]
            # left child
            left_rule = current_rule + ["({} <= {})".format(name, threshold)]
            tree_dfs(inner_tree.children_left[node_id], left_rule)
            # right child
            right_rule = current_rule + ["({} > {})".format(name, threshold)]
            tree_dfs(inner_tree.children_right[node_id], right_rule)
        else:  # leaf
            dist = inner_tree.value[node_id][0]
            dist = dist / dist.sum()
            max_idx = dist.argmax()
            if len(current_rule) == 0:
                rule_string = "ALL"
            else:
                rule_string = " and ".join(current_rule)
            # register new rule to dictionary
            selected_class = classes[max_idx]
            class_probability = dist[max_idx]
            class_rules = class_rules_dict.get(selected_class, [])
            class_rules.append((rule_string, class_probability))
            class_rules_dict[selected_class] = class_rules

    tree_dfs()  # start from root, node_id = 0
    return class_rules_dict


def cluster_report(data: pd.DataFrame, clusters, min_samples_leaf=50, pruning_level=0.01):
    # Create Model
    tree = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, ccp_alpha=pruning_level)
    tree.fit(data, clusters)

    # Generate Report
    feature_names = data.columns
    class_rule_dict = get_class_rules(tree, feature_names)

    report_class_list = []
    for class_name in class_rule_dict.keys():
        rule_list = class_rule_dict[class_name]
        combined_string = ""
        for rule in rule_list:
            combined_string += "[{}] {}\n\n".format(rule[1], rule[0])
        report_class_list.append((class_name, combined_string))

    cluster_instance_df = pd.Series(clusters).value_counts().reset_index()
    cluster_instance_df.columns = ['class_name', 'instance_count']
    report_df = pd.DataFrame(report_class_list, columns=['class_name', 'rule_list'])
    report_df = pd.merge(cluster_instance_df, report_df, on='class_name', how='left')
    pretty_print(report_df.sort_values(by='class_name')[['class_name', 'instance_count', 'rule_list']])

def pretty_print(df):
    print(df.to_string())

if __name__ == '__main__':
    deskriptivnaStatistika()
    fillNaN()
    histogramRaspodjeleSvihKolona()
    boxplotSvihKolona()
    dodajNoveKolone()
    podaci['TYPE_OF_BUYING'] = podaci.apply(odrediTipKupovine, axis=1)
    podaciBezLayeraSkalirani = dodajDummy()
    heatMapZaSveKolone()
    orgPodaci = podaciBezLayeraSkalirani.drop(['TYPE_OF_BUYING'],axis=1)
    izbaciAtribute()
    podaciBezLayeraSkalirani = resavanjeOutleyera()
    kolone=podaciBezLayeraSkalirani.columns
    podaciBezLayeraSkalirani = scale()
    odredjivanjePotrebnogBrojaAtributa()
    skraceniPodaci = izbacivanjeSuvisnogBrojaAtributa()
    odredjivanjeBrojaKlastera()
    kreirajiPrikaziKlastere()