import random

def split_data(data , prob):
    results = [], []
    for row in data:
        results[0 if random.random() < prob else 1].append(row)
    return results

def train_test_split(x, y, test_pct):
    data = zip(x,y)
    train, test = split_data(data, 1-test_pct)
    X_train, y_train = zip(*train)
    X_test, y_test = zip(*test)
    return X_train, X_test, y_train, y_test