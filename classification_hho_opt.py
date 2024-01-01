import random
import numpy as np
import pandas as pd
import math
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class Solution:
    def __init__(self):
        self.startTime = None
        self.endTime = None
        self.executionTime = None
        self.convergence = None
        self.optimizer = None
        self.objfname = None
        self.best = float('inf')  # veya başlangıçta bir büyük sayı olabilir
        self.bestIndividual = None

def HHO(objf, lb, ub, dim, SearchAgents_no, Max_iter):
    if not isinstance(lb, list):
        lb = [lb for _ in range(dim)]
        ub = [ub for _ in range(dim)]
    lb = np.asarray(lb)
    ub = np.asarray(ub)

    # Initialize the locations of Harris' hawks
    X = np.asarray([x * (ub - lb) + lb for x in np.random.uniform(0, 1, (SearchAgents_no, dim))])

    # Initialize convergence
    convergence_curve = np.zeros(Max_iter)
    best_scores = []  # Her iterasyondaki en iyi skorları tutacak liste
    iterations = []  # Her iterasyonu tutacak liste

    s = Solution()

    print("HHO is now tackling  \"" + objf.__name__ + "\"")

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    t = 0  # Loop counter

    # Main loop
    while t < Max_iter:
        for i in range(0, SearchAgents_no):
            fitness = objf(X[i, :])

            if fitness < s.best:
                s.best = fitness
                s.bestIndividual = X[i, :]

        convergence_curve[t] = s.best
        iterations.append(t)  # Her iterasyonu listeye ekle

        if (t % 1 == 0):
            print(['At iteration ' + str(t) + ' the best fitness is ' + str(s.best)])

        t = t + 1

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence_curve
    s.optimizer = "HHO"
    s.objfname = objf.__name__

    # Best score - Iterasyon sayısı grafiğini çiz
    plt.plot(iterations, convergence_curve)
    plt.xlabel('Iteration Number')
    plt.ylabel('Best Score')
    plt.title('HHO - Best Score - Iteration Number Curve')
    plt.show()

    # Best score - Iterasyon sayısı grafiğini döndür
    return s



def Levy(dim):
    beta = 1.5
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / (
            math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = 0.01 * np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    zz = np.power(np.absolute(v), (1 / beta))
    step = np.divide(u, zz)
    return step

# Veri setini yükle
df = pd.read_csv("BreastCancerWisconsin.csv")

# Gereksiz sütunları kaldır
df.drop(["id", "Unnamed: 32"], axis=1, inplace=True)

df = df.rename(columns={"diagnosis": "target"})

# 1 : M and 0 : B
df["target"] = [1 if i.strip() == "M" else 0 for i in df.target]

# Veriyi bağımlı (y) ve bağımsız (X) değişkenlere ayırın
X = df.drop('target', axis=1)
y = df['target']

# Eğitim ve test setlerine ayırın
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Objective Function'ı tanımlayın
def classification_objective_function(params):
    n_estimators = int(params[0])
    max_depth = int(params[1])
    min_samples_split = int(params[2])
    min_samples_leaf = int(params[3])

    # Random Forest modelini oluşturun
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                   min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)

    # Modeli eğitin
    model.fit(X_train, y_train)

    # Test seti üzerinde tahmin yapın
    y_pred = model.predict(X_test)

    # Modelin doğruluğunu değerlendirin
    accuracy = accuracy_score(y_test, y_pred)

    # Hedef, optimize edilmek istenen metrik olacaktır (örneğin, doğruluk).
    return 1 - accuracy  # minimize the negative accuracy

# Harris Hawk Optimizasyonu ile Objective Function'ı optimize edin
hh_optimizer = HHO(objf=classification_objective_function, lb=[1, 1, 2, 1], ub=[100, 50, 20, 10],
                   dim=4, SearchAgents_no=50, Max_iter=50)

# Optimize edilmiş parametreleri alın
optimized_params = hh_optimizer.bestIndividual

# Eğer optimizasyon başarılı olduysa
if optimized_params is not None:
    # Optimize edilmiş parametrelerle final modeli oluşturun
    final_model = RandomForestClassifier(n_estimators=int(optimized_params[0]),
                                         max_depth=int(optimized_params[1]),
                                         min_samples_split=int(optimized_params[2]),
                                         min_samples_leaf=int(optimized_params[3]))

    # Final modeli eğitin
    final_model.fit(X_train, y_train)

    # Test seti üzerinde tahmin yapın
    y_pred_final = final_model.predict(X_test)

    # Optimize edilmiş modelin doğruluğunu değerlendirin
    accuracy_final = accuracy_score(y_test, y_pred_final)
    print(f"Optimized Model Accuracy: {accuracy_final}")

else:
    print("Optimization did not converge to a solution. Please check the optimization process.")
