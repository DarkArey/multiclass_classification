import numpy as np

# Функция генерирует объеты для мультиклассовой классификации с двумя признаками
def gen_dataset(classes_numb: int, obj_numb : int, x1_center: float, x2_center: float, deviation: float, scale: float = 0.5):
    feature_numb = 2
    y = []
    x = np.empty((classes_numb, feature_numb, obj_numb))
    
    # Генерация объектов с заданными центрами и отклонениями
    for i in range(classes_numb):
        x[i, 0, :], x[i, 1, :] = np.random.randint(x1_center[i]-deviation[i], x1_center[i]+deviation[i], size=obj_numb), \
            np.random.randint(x2_center[i]-deviation[i], x2_center[i]+deviation[i], size=obj_numb)
        y = np.concatenate((y, np.repeat(i, obj_numb)))

    # Наложение шума на признаки объектов
    noise = np.random.normal(0, scale, (classes_numb, feature_numb, obj_numb))
    x += noise

    # Перераспределение признаков всех объектов в двумерный массив
    reshape_x = np.empty((feature_numb, classes_numb*obj_numb))
    for i in range(classes_numb):
        reshape_x[:, (i*obj_numb) : ((i+1)*obj_numb)] = x[i, :, :]
    x = reshape_x

    # Перемешивание объектов
    shuffle_idx = np.random.choice(
        np.arange(x.shape[1]), x.shape[1], replace=False)
    x[0, :] = x[0, shuffle_idx]
    x[1, :] = x[1, shuffle_idx]
    y = y[shuffle_idx]

    return x, y