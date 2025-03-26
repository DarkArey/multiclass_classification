import numpy as np
from dataclasses import dataclass


class LinearClassifier:
    @dataclass(frozen=True, slots=True)
    class TrainResult:
        loss: float
        gradient: np.ndarray
        y_estimate: np.ndarray
        weights: np.ndarray

    def __init__(self, bias: float, weights: np.ndarray):
        self._bias: float = bias
        self.weights: np.ndarray = weights

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert x.shape[0] == self.weights.size, "Invalid number of parameters. Should be " + self.weights.size - 1
        return self._bias + self.weights @ x
    
    def train(self, x: np.ndarray, y_reference: np.ndarray, ll: float = 1e-2):
        weights = self.weights_with_bias
        y_estimate = self.forward(x)
        x_padded = np.concat([np.ones((1, x.shape[1])), x], 0)

        gradient = self.calc_gradient(ll, weights, y_reference, y_estimate, x_padded)
        loss = self.calc_loss(ll, weights, y_estimate, y_reference)
        
        self._bias -= gradient[0]
        self.weights -= gradient[1:]

        return self.TrainResult(loss=loss, gradient=gradient, y_estimate=y_estimate, weights=weights)
    
    def calc_gradient(self, ll: float, weights: np.ndarray, y_ref: np.ndarray, y_est: np.ndarray, x: np.ndarray) -> np.ndarray:
        valid = y_est * y_ref <= 0 # y<w,x> <= 0
        gradient = 2 * ll * weights + \
            np.sum(-y_ref[valid] * x[:, valid], 1)
        return gradient
    
    def calc_loss(self, ll: float, weights: np.ndarray, y_est: np.ndarray, y_ref: np.ndarray) -> float:
        loss = ll * np.linalg.norm(weights) + \
            np.sum(np.maximum(0, -y_ref * y_est))
        return loss


    @property
    def weights_with_bias(self) -> np.ndarray:
        weights = np.zeros(self.weights.size + 1)
        weights[0] = self._bias
        weights[1:] = self.weights
        return weights



class SupportVectorMachine(LinearClassifier):
    def __init__(self, bias, weights):
        super().__init__(bias, weights)

    def calc_gradient(self, ll, weights, y_ref, y_est, x):
        valid = 1 - y_est * y_ref > 0
        gradient = 2 * ll * weights + \
            np.sum(-y_ref[valid] * x[:, valid], 1)
        return gradient
    
    def calc_loss(self, ll, weights, y_est, y_ref):
        loss = ll * np.linalg.norm(weights) + \
            np.sum(np.maximum(0, 1 - y_ref * y_est))
        return loss



class LogisticRegression(LinearClassifier):
    def __init__(self, bias, weights):
        super().__init__(bias, weights)

    def calc_gradient(self, ll, weights, y_ref, y_est, x):
        M = np.clip((y_ref * y_est), -700, 700) # Ограничение значений
        sigma = 1 / (1 + np.exp(-M))
        gradient = 2 * ll * weights + \
            np.sum(x * ((sigma) * (-y_ref) * np.exp(-M)), 1)

        return gradient

    def calc_loss(self, ll, weights, y_est, y_ref):
        M = np.clip((y_ref * y_est), -700, 700) # Ограничение значений
        loss = ll * np.linalg.norm(weights) + \
            np.sum(np.log(1 + np.exp(-M)))
        return loss
    



class OneVsAllClassifier:
    def __init__(self, classifier, classes_numb: int, bias, weights):
        self.classes_numb = classes_numb

        # Создание классификатора для каждого класса
        self.classifiers = [classifier(bias, weights.copy()) for _ in range(classes_numb)]


    def train(self, x: np.ndarray, y_reference: np.ndarray, ll: float = 1e-2):
        loss = 0

        # Обучение классификаторов
        for i in range(self.classes_numb):
            # Объекты классификатора имеют класс 1, остальные -1
            binary_y = np.where(y_reference == i, 1, -1)
            r = self.classifiers[i].train(x, binary_y, ll)
            loss += r.loss

        return loss / self.classes_numb


    def classify(self, x: np.ndarray) -> np.ndarray:
        scores = np.empty((self.classes_numb, x.shape[1]))

        # Классификация происходит по выбору наиболее уверенного классификатора
        for i in range(self.classes_numb):
            scores[i, :] = self.classifiers[i].forward(x)

        return np.argmax(scores, axis=0)




class AllVsAllClassifier:
    def __init__(self, classifier, classes_numb: int, bias, weights):
        self.classes_numb = classes_numb
        self.classifiers = {}
        
        # Создание классификатора для каждой пары классов
        for i in range(classes_numb):
            for j in range(i + 1, classes_numb):
                self.classifiers[(i, j)] = classifier(bias, weights.copy())


    def train(self, x: np.ndarray, y_reference: np.ndarray, ll: float = 1e-2):
        loss = 0

        # Обучение классификаторов
        for (i, j), classifier in self.classifiers.items():
            # Классификатор учится только на паре классов
            mask = (y_reference == i) | (y_reference == j)
            binary_x = x[:, mask]
            binary_y = np.where(y_reference[mask] == i, 1, -1)
            r = classifier.train(binary_x, binary_y, ll)
            loss += r.loss

        return loss / self.classes_numb


    def classify(self, x: np.ndarray) -> np.ndarray:
        votes = np.zeros((self.classes_numb, x.shape[1]))
        
        # Классификация происходит по числу голосов классификаторов
        for (i, j), classifier in self.classifiers.items():
            binary_res = classifier.forward(x)
            binary_res = np.where(binary_res > 0, i, j)
            for idx, res in enumerate(binary_res):
                votes[res, idx] += 1
        
        # При двух и более максимумов голосов присваивается -1
        max_votes = np.max(votes, axis=0)
        winners = np.where(votes == max_votes, 1, 0).sum(axis=0)
        result = np.argmax(votes, axis=0)
        result[winners > 1] = -1

        return result