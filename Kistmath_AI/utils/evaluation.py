import numpy as np
from utils.tokenization import tokenize_problem, tokenize_calculus_problem

def evaluate_readiness(model, problems, threshold):
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    if model.get_learning_stage() == 'university':
        X = np.array([tokenize_calculus_problem(p.problem) for p in problems])
        y = np.array([p.solution for p in problems])
    else:
        X = np.array([tokenize_problem(p.problem) for p in problems])
        y_real = np.array([p.solution.real for p in problems])
        y_imag = np.array([p.solution.imag for p in problems])
        y = np.column_stack((y_real, y_imag))
    
    # Ensure input shape is correct
    input_shape = model.input_shape[1:]
    X = X.reshape((-1,) + input_shape)
    
    predictions = model.predict(X)
    
    # Ensure predictions and y have the same shape
    if predictions.shape != y.shape:
        predictions = predictions.squeeze()
    
    mse = np.mean(np.square(y - predictions))
    r2 = 1 - (np.sum(np.square(y - predictions)) / np.sum(np.square(y - np.mean(y))))
    
    return r2 > threshold