import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cholesky, det, inv
from scipy.stats import norm


def knn_2d_bin(x, X1, X2, k):
    training_set = []
    for i in range(X1.shape[1]):
        training_set.append((X1[0][i], X1[1][i], 1))
    for i in range(X2.shape[1]):
        training_set.append((X2[0][i], X2[1][i], 2))
    res = np.zeros(x.shape[1])
    for i in range(x.shape[1]):
        xi = x[:, i]
        neighbors = sorted(training_set,
                           key=lambda sample: (xi[0] - sample[0]) ** 2 + (xi[1] - sample[1]) ** 2)[:k]
        res[i] = 1 if (sum([1 if n[2] == 1 else 0 for n in neighbors]) >= len(neighbors) / 2) else 2
    return res


def randncor(n, N, C):
    try:
        A = cholesky(C).T
        m = n
    except np.linalg.LinAlgError:
        m = n - 1
        # Для простоты, в случае ошибки используем усеченную матрицу
        A = cholesky(C[:m, :m]).T

    # генерация матрицы реализаций m*N гауссовских независимых случайных величин
    u = np.random.randn(m, N)

    # получение матрицы реализаций N гауссовских коррелированных векторов размерности m
    x = A @ u
    return x, m


def main():
    np.set_printoptions(suppress=True, precision=6)
    np.random.seed(42)
    # 1. Задание исходных данных
    n = 2
    M = 2  # размерность признакового пространства и число классов
    K = 10000  # количество статистических испытаний

    # Априорные вероятности, математические ожидания и матрицы ковариации классов
    pw = np.array([0.5, 0.5])
    pw = pw / np.sum(pw)

    m = np.array([[2, -3], [1, 10]]).T  # математические ожидания

    # Матрицы ковариации
    C = np.zeros((n, n, M))
    C[:, :, 0] = np.array([[4, -2], [-2, 4]])
    C[:, :, 1] = np.array([[5, 1], [1, 5]])

    # Обратные матрицы ковариации
    C_ = np.zeros_like(C)
    for k in range(M):
        C_[:, :, k] = inv(C[:, :, k])

    # 2. Расчет матриц вероятностей ошибок распознавания
    PIJ = np.zeros((M, M))
    PIJB = np.zeros((M, M))
    l0_ = np.zeros((M, M))

    for i in range(M):
        for j in range(i + 1, M):
            dmij = m[:, i] - m[:, j]
            l0_[i, j] = np.log(pw[j] / pw[i])

            dti = det(C[:, :, i])
            dtj = det(C[:, :, j])

            # Вычисление следов
            trij = np.trace(C_[:, :, j] @ C[:, :, i] - np.eye(n))
            trji = np.trace(np.eye(n) - C_[:, :, i] @ C[:, :, j])

            trij_2 = np.trace((C_[:, :, j] @ C[:, :, i] - np.eye(n)) @
                              (C_[:, :, j] @ C[:, :, i] - np.eye(n)))
            trji_2 = np.trace((np.eye(n) - C_[:, :, i] @ C[:, :, j]) @
                              (np.eye(n) - C_[:, :, i] @ C[:, :, j]))

            # Параметры нормального распределения
            mg1 = 0.5 * (trij + dmij.T @ C_[:, :, i] @ dmij - np.log(dti / dtj))
            Dg1 = 0.5 * trij_2 + dmij.T @ C_[:, :, j] @ C[:, :, i] @ C_[:, :, j] @ dmij

            mg2 = 0.5 * (trji - dmij.T @ C_[:, :, j] @ dmij + np.log(dtj / dti))
            Dg2 = 0.5 * trji_2 + dmij.T @ C_[:, :, i] @ C[:, :, j] @ C_[:, :, i] @ dmij

            sD1 = np.sqrt(Dg1)
            sD2 = np.sqrt(Dg2)

            # Вероятности ошибок
            PIJ[i, j] = norm.cdf(l0_[i, j], mg1, sD1)
            PIJ[j, i] = 1 - norm.cdf(l0_[i, j], mg2, sD2)

            # Расстояние Бхатачария и границы Чернова
            C_avg = (C[:, :, i] / 2 + C[:, :, j] / 2)
            mu2 = (1 / 8) * dmij.T @ inv(C_avg) @ dmij + 0.5 * np.log((dti + dtj) / (2 * np.sqrt(dti * dtj)))

            PIJB[i, j] = np.sqrt(pw[j] / pw[i]) * np.exp(-mu2)
            PIJB[j, i] = np.sqrt(pw[i] / pw[j]) * np.exp(-mu2)

        # Диагональные элементы
        PIJB[i, i] = 1 - np.sum(PIJB[i, :])
        PIJ[i, i] = 1 - np.sum(PIJ[i, :])

    KK = [100, 1000, 2000, 5000, 10000, 20000, 50000, 100000]

    # 3. Тестирование алгоритма методом статистических испытаний
    Pc_ = np.zeros((len(KK), M, M))  # экспериментальная матрица вероятностей ошибок
    for ki in range(len(KK)):
        K = KK[ki]
        for k in range(K):  # цикл по числу испытаний
            for i in range(M):  # цикл по классам
                # генерация образа i-го класса
                x, _ = randncor(n, 1, C[:, :, i])
                x = x + m[:, i].reshape(-1, 1)

                # вычисление значения разделяющих функций
                u = np.zeros(M)
                for j in range(M):
                    x_diff = x.flatten() - m[:, j]
                    u[j] = (-0.5 * x_diff.T @ C_[:, :, j] @ x_diff -
                            0.5 * np.log(det(C[:, :, j])) + np.log(pw[j]))

                # определение максимума
                iai = np.argmax(u)
                Pc_[ki, i, iai] += 1  # фиксация результата распознавания

        Pc_[ki] = Pc_[ki] / K

    for ki in range(len(KK)):
        K = KK[ki]
        for k in range(K):  # цикл по числу испытаний
            for i in range(M):  # цикл по классам
                # генерация образа i-го класса
                x, _ = randncor(n, 1, C[:, :, i])
                x = x + m[:, i].reshape(-1, 1)

                # вычисление значения разделяющих функций
                u = np.zeros(M)
                for j in range(M):
                    x_diff = x.flatten() - m[:, j]
                    u[j] = (-0.5 * x_diff.T @ C_[:, :, j] @ x_diff -
                            0.5 * np.log(det(C[:, :, j])) + np.log(pw[j]))

                # определение максимума
                iai = np.argmax(u)
                Pc_[ki, i, iai] += 1  # фиксация результата распознавания

        Pc_[ki] = Pc_[ki] / K

    Pc_knn = np.zeros((len(KK), M, M))  # экспериментальная матрица вероятностей ошибок
    for ki in range(len(KK)):
        K = KK[ki]

        X1 = randncor(2, int(K * pw[0]), C[:, :, 0])[0]
        X1 = X1 + m[:, 0].reshape(-1, 1)
        X2 = randncor(2, int(K * pw[1]), C[:, :, 1])[0]
        X2 = X2 + m[:, 1].reshape(-1, 1)

        kn = int(K ** 0.6)
        if (kn % 2 == 0):
            kn += 1

        Nsamples_class = 1000
        x1 = randncor(2, Nsamples_class, C[:, :, 0])[0]
        x1 = x1 + m[:, 0].reshape(-1, 1)
        x2 = randncor(2, Nsamples_class, C[:, :, 1])[0]
        x2 = x2 + m[:, 1].reshape(-1, 1)

        res1 = knn_2d_bin(x1, X1, X2, kn)
        res2 = knn_2d_bin(x2, X1, X2, kn)
        res1_right = np.sum(res1 == 1)
        res2_right = np.sum(res2 == 2)
        Pc_knn[ki][0][0] = res1_right
        Pc_knn[ki][1][1] = res2_right
        Pc_knn[ki][0][1] = Nsamples_class - res1_right
        Pc_knn[ki][1][0] = Nsamples_class - res2_right

    Pc_knn /= Nsamples_class

    plt.subplot(1, 2, 1)

    plt.plot(KK, Pc_[:, 0, 1], label="ошибка 1 рода эксперимент")
    plt.axhline(PIJ[0, 1], label="ошибка 1 рода теория", color='green', linestyle='--')
    plt.axhline(PIJB[0, 1], label="ошибка 1 рода Чернов", color='blue', linestyle='--')
    plt.plot(KK, Pc_knn[:, 0, 1], label="ошибка 1 рода knn")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(KK, Pc_knn[:, 1, 0], label="ошибка 2 рода knn")
    plt.plot(KK, Pc_[:, 1, 0], label="ошибка 2 рода эксперимент")
    plt.axhline(PIJ[1, 0], label="ошибка 2 рода теория", color='green', linestyle='--')
    plt.axhline(PIJB[1, 0], label="ошибка 2 рода Чернов", color='blue', linestyle='--')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
