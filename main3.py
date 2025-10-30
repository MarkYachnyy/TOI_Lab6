from typing import List

import matplotlib.pyplot as plt
import numpy as np

def randncor(n: int, N: int, C: np.ndarray) -> np.ndarray:
    """
    Функция для генерации коррелированных гауссовских случайных величин
    Аналог функции randncor из MATLAB
    """
    # Разложение Холецкого для получения матрицы преобразования
    L = np.linalg.cholesky(C)
    # Генерация некоррелированных гауссовских величин
    Z = np.random.randn(n, N)
    # Преобразование для получения коррелированных величин
    X = L @ Z
    return X


def gen(n: int, M: int, NN: List[int], L: List[int], ps: List[np.ndarray],
        mM: List[np.ndarray], D: List[np.ndarray], ro: List[np.ndarray],
        vis: int) -> List[np.ndarray]:
    """
    Функция для генерации обучающих и тестовых выборок образов
    на основе смесей гауссовских распределений

    Параметры:
    n - размерность признакового пространства
    M - число классов образов
    NN - список, содержащий объемы генерируемых выборок классов
    L - список, содержащий количество компонентов в смесях классов
    ps - список, содержащий вероятности (веса) компонентов смесей классов
    mM - список, содержащий математические ожидания ГСВ компонентов смесей
    D - список, содержащий значения дисперсии ГСВ
    ro - список, содержащий значения коэффициентов корреляции ГСВ в смесях
    vis – ключ для выполнения визуализации областей локализации данных

    Возвращает:
    XN - список, содержащий генерируемые выборки образов различных классов
    """

    N_total = sum(NN)  # общее количество генерируемых выборок
    XN = [None] * M  # инициализация списка для выборок

    # Организация цикла генерации выборок по номерам классов
    for k in range(M):
        XNk = np.zeros((n, NN[k]))
        mMk = mM[k]
        psk = ps[k]
        Dk = D[k]
        rok = ro[k]

        for i in range(NN[k]):  # генерация выборок
            u = np.random.rand()  # определение индекса принадлежности к компоненту смеси
            a = 0
            t = L[k] - 1  # индекс по умолчанию (0-based)

            # Определение компонента смеси
            for j in range(L[k]):
                b = a + psk[j]
                if a <= u < b:
                    t = j
                    break
                a = b

            # Расчет матрицы ковариации ГСВ компонента смеси
            C = np.zeros((n, n))
            for ic in range(n):
                for jc in range(n):
                    C[ic, jc] = Dk[t] * (rok[t] ** abs(ic - jc))

            # Генерация коррелированной гауссовской величины
            XNk[:, i] = randncor(n, 1, C).flatten() + mMk[:, t]

        XN[k] = XNk

    M_ = min(M, 4)  # для визуализации ограничиваем 4 классами

    # Контрольная визуализация полученных выборок для двумерного вектора признаков и M<5
    if vis == 1 and n >= 2:
        plt.figure(figsize=(10, 8))
        plt.grid(True)

        markers = ['ro', 'k+', 'b^', 'g*']

        for k in range(M_):
            XNk = XN[k]
            marker_style = markers[k]
            color = marker_style[0]
            marker = marker_style[1]

            plt.plot(XNk[0, :], XNk[1, :], marker=marker, color=color,
                     linestyle='None', markersize=6, label=f'Class {k + 1}')

        plt.xlabel('x1', fontfamily='monospace')
        plt.ylabel('x2', fontfamily='monospace')
        plt.title('Пространственная локализация генерируемых образов',
                  fontfamily='monospace', fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return XN


# Контрольный пример: "заплетенные восьмерки"
if __name__ == "__main__":
    n = 2
    M = 2
    NN = [1000, 1000]
    L = [2, 2]
    ps = [np.array([0.5, 0.5]), np.array([0.5, 0.5])]
    D = [np.array([1.0, 1.0]), np.array([1.0, 1.0])]
    ro = [np.array([0.5, 0.5]), np.array([-0.5, -0.5])]

    dm = 4
    mM = [
        np.array([[0, dm], [0, dm]]),  # Class 1
        np.array([[dm, 0], [0, dm]])  # Class 2
    ]

    vis = 1

    # Генерация данных
    XN = gen(n, M, NN, L, ps, mM, D, ro, vis)

    # Проверка размерностей
    for i, x in enumerate(XN):
        print(f"Class {i + 1}: shape {x.shape}")


def vknn(xx, XN, k):
    x=xx.reshape(-1)
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
    # t = np.dot(diff.T, diff) if n > 1 else (diff ** 2)
    return np.sum(np.exp(-np.sum(diff ** 2, axis=0) / (2 * h ** 2))) / (N * (2 * np.pi * h ** 2) ** (n / 2))


def main():
    # 1. Задание исходных данных
    n = 2
    M = 2
    H = 1
    K = 1000
    pw = np.ones(M) / M

    # Исходные данные для генерации обучающих выборок
    L = [2, 2]
    dm = 2
    DM = 1

    # Инициализация параметров смесей
    ps = [np.ones(L_i) / L_i for L_i in L]
    mM = [np.zeros((n, L_i)) + (i) * DM for i, L_i in enumerate(L)]
    D = [np.ones(L_i) for L_i in L]

    # Коэффициенты корреляции
    ro = []
    for i in range(M):
        if n == 2:
            ro.append(2 * np.random.rand(L[i]) - 1)
        else:
            ro.append(np.zeros(L[i]))

    # Настройка математических ожиданий для рис.5.13
    mM[0] = np.array([[0, dm], [0, dm]])
    mM[1] = np.array([[dm, 0], [0, dm]]) + DM

    # Параметры оценок
    kl_kernel = 11
    r = 0.5
    gm = 0.25

    # 2. Генерация обучающих данных в цикле с переменным объемом выборки
    Nn = [10, 20, 30, 40, 50, 100, 150, 200, 250]
    ln = len(Nn)

    Esth1 = np.zeros(ln)
    Esth2 = np.zeros(ln)
    Esex1 = np.zeros(ln)
    Esex2 = np.zeros(ln)

    for nn in range(ln):
        NN = [Nn[nn]] * M
        N = Nn[nn]
        h_N = N ** (-r / n)  # размеры окна Парзена
        k = 2 * round(N ** gm) + 1  # k - число ближайших соседей

        for h in range(H):
            # Генерация обучающих выборок
            XN = gen(n, M, NN, L, ps, mM, D, ro, h)

            # 3. Определение вероятностей ошибок методом скользящего контроля
            Pc1 = np.zeros((M, M))
            Pc2 = np.zeros((M, M))

            for i in range(M):
                XNi = XN[i]
                XNi_ = np.zeros((n, N - 1))
                indi = [idx for idx in range(M) if idx != i]

                for j in range(N):
                    x = XNi[:, j]
                    # Создание обучающей выборки без j-го элемента
                    if j > 0:
                        XNi_[:, :j] = XNi[:, :j]
                    if j < N - 1:
                        XNi_[:, j:] = XNi[:, j + 1:]

                    p1_ = np.zeros(M)
                    p2_ = np.zeros(M)

                    # Оценка Парзена для i-го класса (без j-го элемента)
                    #p1_[i] = vkernel(x, XNi_, h_N, kl_kernel)
                    # Оценка k-ближайших соседей для i-го класса (без j-го элемента)
                    p2_[i] = vknn(x, XNi_, k)

                    # Оценки для других классов
                    for t, ij in enumerate(indi):
                        p1_[ij] = vkernel(x, XN[ij], h_N, kl_kernel)
                        p2_[ij] = vknn(x, XN[ij], k)

                    # Определение максимумов
                    iai1 = np.argmax(p1_)
                    iai2 = np.argmax(p2_)

                    Pc1[i, iai1] += 1
                    Pc2[i, iai2] += 1

                Pc1[i, :] /= N
                Pc2[i, :] /= N

            Esth1[nn] += (1 - np.sum(pw * np.diag(Pc1)))
            Esth2[nn] += (1 - np.sum(pw * np.diag(Pc2)))

            # 4. Тестирование алгоритмов методом статистических испытаний
            Pc1_ = np.zeros((M, M))
            Pc2_ = np.zeros((M, M))

            # Генерация тестирующей выборки
            X = gen(n, M, [K] * M, L, ps, mM, D, ro, 0)

            for i in range(M):
                xi = X[i]
                p1x = np.zeros((M, K))
                p2x = np.zeros((M, K))

                for j in range(M):
                    #p1x[j, :] = vkernel(xi, XN[j], h_N, kl_kernel)
                    p2x[j, :] = vknn(xi, XN[j], k)

                mai1 = np.argmax(p1x, axis=0)
                mai2 = np.argmax(p2x, axis=0)

                ni1 = np.sum(mai1 == i)
                ni2 = np.sum(mai2 == i)

                Pc1_[i, i] = ni1 / K
                Pc2_[i, i] = ni2 / K

            Esex1[nn] += (1 - np.sum(pw * np.diag(Pc1_)))
            Esex2[nn] += (1 - np.sum(pw * np.diag(Pc2_)))

    # Усреднение по статистическим испытаниям
    Esth1 /= H
    Esth2 /= H
    Esex1 /= H
    Esex2 /= H

    # 5. Визуализация зависимостей вероятностей ошибок
    plt.figure(figsize=(12, 8))
    plt.grid(True)

    ms = max([np.max(Esth1), np.max(Esth2), np.max(Esex1), np.max(Esex2)])

    plt.axis([Nn[0], Nn[-1], 0, ms + 0.001])

    plt.plot(Nn, Esth1, '-b', linewidth=1.0, label='Esth1')
    plt.plot(Nn, Esth2, '-r', linewidth=1.0, label='Esth2')
    plt.plot(Nn, Esex1, '--ok', linewidth=1.0, label='Esex1')
    plt.plot(Nn, Esex2, '--^k', linewidth=1.0, label='Esex2')

    plt.title('Суммарная вероятность ошибки', fontsize=14)
    plt.xlabel('N', fontsize=14)
    plt.ylabel('Es', fontsize=14)

    # Формирование информационной строки
    strv = (f"pw={pw} n={n} L={L} dm={dm} DM={DM} H={H}")

    plt.text(Nn[2] + 1, 0.5 * ms, strv,
             horizontalalignment='left',
             bbox=dict(facecolor='#cccccc', alpha=0.8),
             fontsize=12)

    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
