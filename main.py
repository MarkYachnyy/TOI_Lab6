import numpy as np
from scipy.stats import multivariate_normal, norm
import matplotlib.pyplot as plt

def randncor(n, N, C):
    """Генерирует N коррелированных гауссовских выборок размерности n с ковариационной матрицей C"""
    return np.random.multivariate_normal(np.zeros(n), C, N).T

def vknn(x, XN, k):
    """Оценка плотности методом K ближайших соседей для двумерного случая"""
    N = XN.shape[1]
    p_ = np.zeros(x.shape[1])
    for idx in range(x.shape[1]):
        dist = np.linalg.norm(XN - x[:, idx:idx + 1], axis=0)
        sorted_dist = np.sort(dist)
        V = np.pi * sorted_dist[k - 1] ** 2  # объем окрестности (площадь круга)
        p_[idx] = k / (N * V) if V > 0 else 0
    return p_

def vkernel(x, XN, h, kl_kernel):
    """Парзеновская оценка плотности (ядро с Гауссовым окном)"""
    n, N = XN.shape
    diff = XN - x.reshape(-1, 1)
    #t = np.dot(diff.T, diff) if n > 1 else (diff ** 2)
    return np.sum(np.exp(-np.sum(diff ** 2, axis=0) / (2 * h ** 2))) / (N * (2 * np.pi * h ** 2) ** (n / 2))


# ---- Основной код ----

# Разные значения объема выборки
KK = [10000]
err1c2 = np.zeros(len(KK))
err2c2 = np.zeros(len(KK))

# 1. Параметры задачи
n = 2
M = 2
dm = 2.0
g = 0.6
C = np.zeros((n, n, M))
C_ = np.zeros((n, n, M))
pw = np.array([0.5, 0.5, 0.5])
pw = pw / np.sum(pw)
m = np.array([[2, 1], [-3, 10]])
C[:, :, 0] = np.array([[4, -2], [-2, 4]])
C[:, :, 1] = np.array([[5, 1], [1, 5]])
for k in range(M):
    C_[:, :, k] = np.linalg.inv(C[:, :, k])

for tt, K in enumerate(KK):

    # – Число образов каждого класса
    Ks = (K * pw).astype(int)
    Ks[-1] = K - np.sum(Ks[:-1])

    # 1.1 Генерация обучающих выборок
    XN = []
    for i in range(M):
        XNi = m[:, [i]] + randncor(n, Ks[i], C[:, :, i])
        XN.append(XNi)

    # 2. Расчет матриц вероятностей ошибок распознавания
    PIJ = np.zeros((M, M))
    PIJB = np.zeros((M, M))
    l0_ = np.zeros((M, M))
    for i in range(M):
        for j in range(i + 1, M):
            dmij = m[:, i] - m[:, j]
            l0_[i, j] = np.log(pw[j] / pw[i])
            dti = np.linalg.det(C[:, :, i])
            dtj = np.linalg.det(C[:, :, j])
            trij = np.trace(np.dot(C_[:, :, j], C[:, :, i]) - np.eye(n))
            trji = np.trace(np.eye(n) - np.dot(C_[:, :, i], C[:, :, j]))
            mg1 = 0.5 * (trij + dmij.T @ C_[:, :, j] @ dmij - np.log(dti / dtj))
            Dg1 = 0.5 * trij ** 2 + dmij.T @ C_[:, :, j] @ C[:, :, i] @ C_[:, :, j] @ dmij
            mg2 = 0.5 * (trji - dmij.T @ C_[:, :, i] @ dmij + np.log(dtj / dti))
            Dg2 = 0.5 * trji ** 2 + dmij.T @ C_[:, :, i] @ C[:, :, j] @ C_[:, :, i] @ dmij
            sD1 = np.sqrt(Dg1)
            sD2 = np.sqrt(Dg2)
            PIJ[i, j] = norm.cdf(l0_[i, j], mg1, sD1)
            PIJ[j, i] = 1 - norm.cdf(l0_[i, j], mg2, sD2)
            mu2 = (1 / 8) * dmij.T @ np.linalg.inv(0.5 * (C[:, :, i] + C[:, :, j])) @ dmij + 0.5 * np.log(
                (dti + dtj) / (2 * np.sqrt(dti * dtj)))
            PIJB[i, j] = np.sqrt(pw[j] / pw[i]) * np.exp(-mu2)
            PIJB[j, i] = np.sqrt(pw[i] / pw[j]) * np.exp(-mu2)
        PIJB[i, i] = 1 - np.sum(PIJB[i, :])
        PIJ[i, i] = 1 - np.sum(PIJ[i, :])

    # 2.1 Определение вероятностей ошибок методом скользящего контроля
    r = 0.5
    kl_kernel = 11
    Pc1 = np.zeros((M, M))
    p1_ = np.zeros(M)
    for i in range(M):
        N = Ks[i]
        XNi = XN[i]
        indi = [j for j in range(M) if j != i]
        for j in range(N):
            knn = int(N ** g)
            x = XNi[:, j]
            mask = np.ones(XNi.shape[1], dtype=bool)
            mask[j] = 0
            XNi_ = XNi[:, mask]
            h_N = N ** (-r / n)
            p1_[i] = vknn(np.reshape(x, (2,1)), XNi_, knn)
            for t, ij in enumerate(indi):
                h_N_ij = Ks[ij] ** (-r / n)
                p1_[ij] = vknn(np.reshape(x, (2,1)), XN[ij], knn)
            iai1 = np.argmax(p1_)
            Pc1[i, iai1] += 1
        Pc1[i, :] /= N

    # 3. Тестирование методом статистических испытаний
    Pcv = np.zeros((M, M))
    Pc_ = np.zeros((M, M))
    p = np.zeros(M)
    for k in range(K):
        for i in range(M):
            x = m[:, i] + randncor(n, 1, C[:, :, i]).flatten()
            u = np.zeros(M)
            for j in range(M):
                u[j] = -0.5 * (x - m[:, j]).T @ C_[:, :, j] @ (x - m[:, j]) \
                       - 0.5 * np.log(np.linalg.det(C[:, :, j])) + np.log(pw[j])
                h_N = Ks[j] ** (-r / n)
                p[j] = vknn(x.reshape((2,1)), XN[j], int(round(K**g)))
            iai = np.argmax(u)
            Pc_[i, iai] += 1
            iaip = np.argmax(p)
            Pcv[i, iaip] += 1
    Pc_ /= K
    Pcv /= K

    # Ошибки для третьего класса
    err1c2[tt] = Pcv[1, 0]  # первый род: ошибки, присвоенные не второму (строка)
    err2c2[tt] = Pcv[0, 1]  # второй род: другие классы приняли за второй (столбец)


print("Теоретическая матрица вероятностей ошибок:\n", PIJ)
print("Матрица ошибок по методу скользящего контроля:\n", Pc1)
print("Матрица ошибок (граница Чернова):\n", PIJB)
print("Экспериментальная матрица ошибок (Гауссовский классификатор):\n", Pc_)
print("Экспериментальная матрица ошибок (оценки Knn):\n", Pcv)

x1 = np.arange(-15, 15.1, 0.5)
x2 = np.arange(-15, 15.1, 0.5)
X1, X2 = np.meshgrid(x1, x2)
x_flat = np.vstack([X1.ravel(), X2.ravel()])

p = np.zeros((2, x_flat.shape[1]))
p_est = np.zeros((2, x_flat.shape[1]))

for cl in range(2):
    N = 5000
    XN_opt = np.random.multivariate_normal(m[:, cl], C[:, :, cl], N).T
    p_est[cl] = vknn(x_flat, XN_opt, int(N ** 0.6))
    p[cl] = multivariate_normal.pdf(x_flat.T, mean=m[:, cl], cov = C[:, :, cl])

fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X1, X2, p[0].reshape(len(x1), len(x2)), cmap='viridis', alpha=0.8)
ax1.plot_surface(X1, X2, p[1].reshape(len(x1), len(x2)), cmap='viridis', alpha=0.8)
ax1.set_title(f'Истинная плотность', fontsize=12)
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('p')
plt.tight_layout()

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X1, X2, p_est[0].reshape(len(x1), len(x2)), cmap='viridis', alpha=0.8)
ax2.plot_surface(X1, X2, p_est[1].reshape(len(x1), len(x2)), cmap='viridis', alpha=0.8)
ax2.set_title(f'Экспериментальная плотность', fontsize=12)
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_zlabel('p~')
plt.tight_layout()
plt.show()