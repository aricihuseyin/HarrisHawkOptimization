import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import math


iris = datasets.load_iris()
X = iris.data
y = iris.target

# Veri setini eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# SVM modelini değerlendiren ve performans metriklerini hesaplayan bir fonksiyon
def evaluate_svm_model(C, gamma, X_train, y_train, X_test, y_test):
    model = SVC(C=C, gamma=gamma, kernel='rbf')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    return 1 - acc, 1 - f1  # Hedef fonksiyonun minimize edilmesi için negatif performans metrikleri kullanılır


# HHO optimizasyonu
def hho_optimization(obj_func, n_iterations, n_hawks, pa, fl, fh, X_train, y_train, X_test, y_test):
    C = np.random.uniform(0.1, 100)
    gamma = np.random.uniform(0.1, 10)

    Rabbit_Location = np.mean(X_train, axis=0)  # Burada Rabbit_Location'ı tanımlıyoruz

    for _ in range(n_iterations):
        current_fitness = obj_func(C, gamma, X_train, y_train, X_test, y_test)

        E1 = 2 * (1 - (_ / n_iterations))
        E0 = 2 * np.random.random() - 1
        Escaping_Energy = E1 * E0

        if abs(Escaping_Energy) >= 1:
            q = np.random.random()
            rand_Hawk_index = np.floor(n_hawks * np.random.random())
            X_rand = np.copy(X_train)
            if q < 0.5:
                X_train = X_rand - np.random.random() * np.abs(X_rand - 2 * np.random.random() * X_rand)
            elif q >= 0.5:
                X_train = Rabbit_Location - X_rand - np.random.random() * (
                            (np.max(X_rand, axis=0) - np.min(X_rand, axis=0)) * np.random.random() + np.min(X_rand, axis=0))
        elif abs(Escaping_Energy) < 1:
            r = np.random.random()
            if r >= 0.5 and abs(Escaping_Energy) < 0.5:
                X_train = Rabbit_Location - Escaping_Energy * np.abs(Rabbit_Location - X_train)
            if r >= 0.5 and abs(Escaping_Energy) >= 0.5:
                Jump_strength = 2 * (1 - np.random.random())
                X_train = Rabbit_Location - X_train - Escaping_Energy * np.abs(Jump_strength * Rabbit_Location - X_train)
            if r < 0.5 and abs(Escaping_Energy) >= 0.5:
                Jump_strength = 2 * (1 - np.random.random())
                X1 = Rabbit_Location - Escaping_Energy * np.abs(Jump_strength * Rabbit_Location - X_train)
                X1 = np.clip(X1, 0.1, 100)
                if obj_func(C, gamma, X_train, y_train, X_test, y_test) < current_fitness:
                    X_train = np.copy(X1)
                else:
                    Jump_strength = 2 * (1 - np.random.random())
                    X2 = Rabbit_Location - Escaping_Energy * np.abs(Jump_strength * Rabbit_Location - X_train) + np.multiply(
                        np.random.randn(X_train.shape[1]), Levy(X_train.shape[1]))
                    X2 = np.clip(X2, 0.1, 100)
                    if obj_func(C, gamma, X_train, y_train, X_test, y_test) < current_fitness:
                        X_train = np.copy(X2)
            if r < 0.5 and abs(Escaping_Energy) < 0.5:
                Jump_strength = 2 * (1 - np.random.random())
                X1 = Rabbit_Location - Escaping_Energy * np.abs(Jump_strength * Rabbit_Location - np.mean(X_train, axis=0))
                X1 = np.clip(X1, 0.1, 100)
                if obj_func(C, gamma, X_train, y_train, X_test, y_test) < current_fitness:
                    X_train = np.copy(X1)
                else:
                    Jump_strength = 2 * (1 - np.random.random())
                    X2 = Rabbit_Location - Escaping_Energy * np.abs(
                        Jump_strength * Rabbit_Location - np.mean(X_train, axis=0)) + np.multiply(np.random.randn(X_train.shape[1]),
                                                                                            Levy(X_train.shape[1]))
                    X2 = np.clip(X2, 0.1, 100)
                    if obj_func(C, gamma, X_train, y_train, X_test, y_test) < current_fitness:
                        X_train = np.copy(X2)

    return C, gamma




# Levy uçuşu
def Levy(dim):
    beta = 1.5
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / (
        math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = 0.01 * np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    zz = np.power(np.abs(v), (1 / beta))
    step = np.divide(u, zz)
    return step


# PSO optimizasyonu
def pso_optimization(obj_func, n_iterations, n_particles, X_train, y_train, X_test, y_test):
    C = np.random.uniform(0.1, 100, size=n_particles)
    gamma = np.random.uniform(0.1, 10, size=n_particles)

    velocities = np.zeros((n_particles, 2))
    personal_best = np.copy(C), np.copy(gamma)
    global_best = min([(obj_func(c, g, X_train, y_train, X_test, y_test), (c, g)) for c, g in zip(C, gamma)],
                      key=lambda x: x[0])

    for _ in range(n_iterations):
        for i in range(n_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            velocities[i, 0] = 0.5 * velocities[i, 0] + 2 * r1 * (personal_best[0][i] - C[i]) + 2 * r2 * (
                        global_best[1][0] - C[i])
            velocities[i, 1] = 0.5 * velocities[i, 1] + 2 * r1 * (personal_best[1][i] - gamma[i]) + 2 * r2 * (
                        global_best[1][1] - gamma[i])
            C[i] += velocities[i, 0]
            gamma[i] += velocities[i, 1]

            # Hiperparametre sınırlarını kontrol etme
            C[i] = np.clip(C[i], 0.1, 100)
            gamma[i] = np.clip(gamma[i], 0.1, 10)

        # En iyi konumu güncelleme
        current_best = min([(obj_func(c, g, X_train, y_train, X_test, y_test), (c, g)) for c, g in zip(C, gamma)],
                           key=lambda x: x[0])
        if current_best[0] < global_best[0]:
            global_best = current_best

        # Bireysel en iyiyi güncelleme
        for i in range(n_particles):
            if current_best[0] < obj_func(personal_best[0][i], personal_best[1][i], X_train, y_train, X_test, y_test):
                personal_best[0][i] = np.copy(C[i])
                personal_best[1][i] = np.copy(gamma[i])

    return global_best[1]


# HHO ve PSO ile SVM'yi optimize etme
best_params_hho = hho_optimization(evaluate_svm_model, n_iterations=50, n_hawks=20, pa=0.8, fl=2, fh=3, X_train=X_train,
                                   y_train=y_train, X_test=X_test, y_test=y_test)
best_params_pso = pso_optimization(evaluate_svm_model, n_iterations=50, n_particles=20, X_train=X_train,
                                   y_train=y_train, X_test=X_test, y_test=y_test)

# En iyi parametrelerle SVM modellerini oluşturma
best_model_hho = SVC(C=best_params_hho[0], gamma=best_params_hho[1], kernel='rbf')
best_model_pso = SVC(C=best_params_pso[0], gamma=best_params_pso[1], kernel='rbf')

# En iyi parametrelerle SVM modellerini eğitme
best_model_hho.fit(X_train, y_train)
best_model_pso.fit(X_train, y_train)

# Test seti üzerinde performansı değerlendirme
y_pred_hho = best_model_hho.predict(X_test)
y_pred_pso = best_model_pso.predict(X_test)

accuracy_hho = accuracy_score(y_test, y_pred_hho)
f1_hho = f1_score(y_test, y_pred_hho, average='weighted')

accuracy_pso = accuracy_score(y_test, y_pred_pso)
f1_pso = f1_score(y_test, y_pred_pso, average='weighted')

# Sonuçları yazdırma
print("HHO ile Optimize Edilmiş SVM:")
print("En iyi parametreler: C={}, gamma={}".format(best_params_hho[0], best_params_hho[1]))
print("Accuracy:", accuracy_hho)
print("F1 Score:", f1_hho)
print("\nPSO ile Optimize Edilmiş SVM:")
print("En iyi parametreler: C={}, gamma={}".format(best_params_pso[0], best_params_pso[1]))
print("Accuracy:", accuracy_pso)
print("F1 Score:", f1_pso)
