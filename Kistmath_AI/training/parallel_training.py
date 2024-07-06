import multiprocessing
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from utils.tokenization import tokenize_problem, tokenize_calculus_problem
from models.kistmat_ai import Kistmat_AI

def train_fold(fold_data):
    model_config, model_weights, train_problems, val_problems, epochs = fold_data
    model = Kistmat_AI.from_config(model_config)
    model.set_weights(model_weights)
    model.compile(optimizer='adam', loss='mse', metrics=['mae']) 
    
    if model.get_learning_stage() == 'university':
        X_train = np.array([tokenize_calculus_problem(p.problem) for p in train_problems])
        y_train = np.array([p.solution for p in train_problems])
    else:
        X_train = np.array([tokenize_problem(p.problem) for p in train_problems])
        y_train_real = np.array([p.solution.real for p in train_problems])
        y_train_imag = np.array([p.solution.imag for p in train_problems])
        y_train = np.column_stack((y_train_real, y_train_imag))
    
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=64, validation_split=0.2, verbose=0)
    
    return {'history': history.history, 'weights': model.get_weights()}

def parallel_train_model(model, problems, epochs=10, n_folds=3):
    kf = KFold(n_splits=n_folds)
    fold_data = []
    
    model_config = model.get_config()
    model_weights = model.get_weights()
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    for train_index, val_index in kf.split(problems):
        train_problems = [problems[i] for i in train_index]
        val_problems = [problems[i] for i in val_index]
        fold_data.append((model_config, model_weights, train_problems, val_problems, epochs))
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        fold_histories = pool.map(train_fold, fold_data)
    
    return fold_histories

def reinforce_single(args):
    model, problem, prediction, true_solution = args
    if model.get_learning_stage() == 'university':
        inputs = tf.convert_to_tensor([tokenize_calculus_problem(problem.problem)])
        true_solution_tensor = tf.constant([[true_solution]], dtype=tf.float32)
    else:
        inputs = tf.convert_to_tensor([tokenize_problem(problem.problem)])
        true_solution_tensor = tf.constant([[true_solution.real, true_solution.imag]], dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        predicted = model(inputs, training=True)
        loss = tf.reduce_mean(tf.square(true_solution_tensor - predicted))
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss.numpy()

def parallel_reinforce_learning(model, problems, predictions, true_solutions, learning_rate=0.01):
    reinforce_data = [(model, problem, prediction, true_solution) 
                      for problem, prediction, true_solution in zip(problems, predictions, true_solutions)]
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        losses = pool.map(reinforce_single, reinforce_data)
    
    return losses