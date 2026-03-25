import numpy as np

def _sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    X = np.array(X)
    y = np.array(y)
    
    n_samples, n_features = X.shape 

    w = np.zeros(n_features)
    b = 0.0

    for step in range(steps):
        z = np.dot(X, w) + b
        p = _sigmoid(z)

        error = p - y 
        
        dL_w = (1 / n_samples) * np.dot(X.T, error)
        dL_b = (1 / n_samples) * np.sum(error)

        w -= lr * dL_w
        b -= lr * dL_b

        if step % 100 == 0:
            loss = -np.mean(y * np.log(p + 1e-15) + (1 - y) * np.log(1 - p + 1e-15))
            print(f"Step {step}: Loss {loss:.4f}")

    return w, b