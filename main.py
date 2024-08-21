import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

data, meta = arff.loadarff('Rice_Cammeo_Osmancik.arff')
df = pd.DataFrame(data)
print("dataset original\n")
df.info()
for i in range(1, 2):
    print("iteração ", i)
    df = shuffle(df)
    atributos = df.iloc[:,:-1]
    classes = df.iloc[:,-1]
    atributos_treino,atributos_temp,classes_treino,classes_temp=train_test_split(atributos,classes,test_size=0.5,stratify=classes)
    atributos_validacao,atributos_ttestes,classes_validacao,classes_testes=train_test_split(atributos_temp,classes_temp,test_size=0.5,stratify=classes_temp)
    acc=0.0
    best_naive_bayes = GaussianNB()
    best_naive_bayes.fit(atributos_treino, classes_treino)
    opiniao = best_naive_bayes.predict(atributos_validacao)
    temp_acc = accuracy_score(classes_validacao, opiniao)
    print("validação, a acurácia do nayve_bays foi de ", temp_acc*100, "%")
    best_knn=None
    for peso in ("distance", "uniform"):
        for k in range(1, 51):
            temp = KNeighborsClassifier(n_neighbors=k, weights=peso)
            temp.fit(atributos_treino, classes_treino)
            opiniao = temp.predict(atributos_validacao)
            temp_acc = accuracy_score(classes_validacao, opiniao)
            if(temp_acc>acc):
                print("validação, foi encontrado um melhor conjunto de parâmetros para o KNN:\nK = ", k, "\npeso = ", peso, "\n", "acurácia = ", temp_acc, "%\n")
                best_knn = temp
                acc=temp_acc