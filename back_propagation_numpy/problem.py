from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


def make_problem(samples: int, test_size: float):
    X, y = make_moons(n_samples=samples, noise=0.2, random_state=100)
    return train_test_split(X, y, test_size=test_size, random_state=42)
