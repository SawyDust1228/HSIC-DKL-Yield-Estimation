import numpy as np
import math
import torch

def sort(y, num = 20):
    indeces = range(y.shape[0])
    indexAndValue = list(zip(indeces, y))
    indexAndValue = sorted(indexAndValue, key=lambda x : x[1], reverse=True)
    return list(map(lambda x : x[0], indexAndValue))[:num]

def getAnswer(answer, result):
    sum = 0
    for item in result:
        if item in answer:
            sum += 1
    print(f"[RATE] {sum} / {len(answer)}")
    return sum


def normalize(value):
    value = (value - np.min(value)) / (np.max(value) - np.min(value))
    return value


def toy_function(train_x):
    return torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.04)

def get_yield(y, threshold):
    sum = 0
    for i in range(y.shape[0]):
        if y[i].item() >= threshold:
            sum += 1
    return 1 - sum / y.shape[0]


# def EI(model, X, Xsamples):
