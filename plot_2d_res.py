import numpy as np
import matplotlib.pyplot as plt

# Функция выводит объекты мультиклассовой классификации с двумя признаками
def plot_2d_res(x: float, y: int, classes_numb: int, plot_type: int = 0, alg_type: int = 0):
    assert x.shape[0] == 2, "Invalid number of features. Should be 2"

    idx = []
    # Определение индексов объектов классов
    for i in range(classes_numb):
        idx.append(np.where(y == i))
    if (alg_type == 1):
        no_class_idx = np.where(y == -1)
    
    # Отрисовка кургами или точками
    if (plot_type == 0):
        plt.plot(x[0, idx[0]], x[1, idx[0]], 'bo')
        plt.plot(x[0, idx[1]], x[1, idx[1]], 'ro')
        if (classes_numb > 2):
            plt.plot(x[0, idx[2]], x[1, idx[2]], 'yo')
        if (classes_numb > 3):
            plt.plot(x[0, idx[3]], x[1, idx[3]], 'go')
        if (alg_type == 1):
            plt.plot(x[0, no_class_idx], x[1, no_class_idx], 'bo')
        plt.grid(1)
    elif (plot_type == 1):
        plt.plot(x[0, idx[0]], x[1, idx[0]], 'b.')
        plt.plot(x[0, idx[1]], x[1, idx[1]], 'r.')
        if (classes_numb > 2):
            plt.plot(x[0, idx[2]], x[1, idx[2]], 'y.')
        if (classes_numb > 3):
            plt.plot(x[0, idx[3]], x[1, idx[3]], 'g.')
        if (alg_type == 1):
            plt.plot(x[0, no_class_idx], x[1, no_class_idx], 'k.')
        plt.grid(1)