import numpy as np


def binary_cross_entropy_loss(target, predicted):
    loss = 0

    for i, y in enumerate(target):
        y_hat = predicted[i]
        loss += y*np.log(y_hat) + (1-y)*np.log(1-y_hat)

    return -loss / len(target)


def binary_cross_entropy_loss_np(target, predicted):
    return (target*np.log(predicted) + (1-target)*np.log(1-predicted)).mean()


target = [1, 0, 0, 1]
predicted_good = [0.9, 0.1, 0.2, 0.8]
predicted_bad =  [0.6, 0.7, 0.8, 0.1]

print( binary_cross_entropy_loss(target, predicted_good))
print( binary_cross_entropy_loss(target, predicted_bad))

print( binary_cross_entropy_loss_np(np.array(target), np.array(predicted_good)))
print( binary_cross_entropy_loss_np(np.array(target), np.array(predicted_bad)))

