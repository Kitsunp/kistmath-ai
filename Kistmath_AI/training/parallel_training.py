import multiprocessing
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from utils.tokenization import tokenize_problem, tokenize_calculus_problem
from models.kistmat_ai import Kistmat_AI

def train_fold(fold_data):
    try:
        model_config, model_weights, train_problems, val_problems, epochs = fold_data

        # Recrear el modelo a partir de la configuración
        model = Kistmat_AI.from_config(model_config)
        model.set_weights(model_weights)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        current_stage = model.get_learning_stage()

        # Preparar los datos de entrenamiento
        X_train = np.array([tokenize_problem(p.problem, current_stage) for p in train_problems])
        if current_stage == 'university':
            y_train = np.array([np.abs(p.solution) for p in train_problems])
        else:
            y_train = np.array([p.solution for p in train_problems])

        # Validación de formas
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(f"Discrepancia en el número de muestras: X_train tiene {X_train.shape[0]}, y_train tiene {y_train.shape[0]}")

        print(f"Forma inicial de X_train: {X_train.shape}")
        print(f"Forma inicial de y_train: {y_train.shape}")

        # Asegurar que X_train tiene la forma correcta
        X_train = np.expand_dims(X_train, axis=-1)

        # Asegurar que y_train sea float32
        y_train = y_train.astype(np.float32)

        # Asegurar que y_train tiene la forma correcta
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        print(f"Forma final de X_train: {X_train.shape}")
        print(f"Forma final de y_train: {y_train.shape}")

        # Verificar que X_train y y_train tienen el mismo número de muestras
        assert X_train.shape[0] == y_train.shape[0], (
            f"Discrepancia en el número de muestras: X_train tiene {X_train.shape[0]} muestras, "
            f"y_train tiene {y_train.shape[0]} muestras"
        )

        all_history = []
        # Entrenar por épocas
        for epoch in range(epochs):
            print(f"Entrenando época {epoch + 1}/{epochs}")
            history = model.fit(
                X_train, y_train,
                epochs=1,
                batch_size=64,
                validation_split=0.2,
                verbose=1
            )
            all_history.append(history.history)

            # Guardar pesos después de cada época (opcional)
            model.save_weights(f'temp_weights_epoch_{epoch}.h5')

        # Combinar los historiales de todas las épocas
        combined_history = {k: np.concatenate([h[k] for h in all_history]) for k in all_history[0].keys()}

        # Evaluar el modelo en los datos de validación
        X_val = np.array([tokenize_problem(p.problem, current_stage) for p in val_problems])
        y_val = np.array([p.solution for p in val_problems])

        # Asegurar que X_val tiene la forma correcta
        X_val = np.expand_dims(X_val, axis=-1)

        # Asegurar que y_val tiene la forma correcta
        if y_val.ndim == 1:
            y_val = y_val.reshape(-1, 1)
        elif hasattr(y_val[0], 'real') and hasattr(y_val[0], 'imag'):
            y_val = np.array([[sol.real, sol.imag] for sol in y_val])

        # Evaluar el modelo
        val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
        print(f"Pérdida de validación: {val_loss}, MAE de validación: {val_mae}")

        # Devolver el resultado
        return {
            'history': combined_history, 
            'weights': model.get_weights(),
            'val_loss': val_loss,
            'val_mae': val_mae
        }

    except Exception as e:
        print(f"Error en train_fold: {str(e)}")
        raise

def parallel_train_model(model, problems, epochs=10, n_folds=3):
    kf = KFold(n_splits=n_folds)
    fold_data = []

    model_config = model.get_config()
    model_weights = model.get_weights()
    for train_index, val_index in kf.split(problems):
        train_problems = [problems[i] for i in train_index]
        val_problems = [problems[i] for i in val_index]
        fold_data.append((model_config, model_weights, train_problems, val_problems, epochs))

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        fold_histories = pool.map(train_fold, fold_data)

    return fold_histories

def reinforce_single(args):
    try:
        model, problem, prediction, true_solution = args
        inputs = tf.convert_to_tensor([tokenize_problem(problem.problem, model.get_learning_stage())])
        inputs = tf.reshape(inputs, (-1, model.input_shape[0]))  # Asegúrate de que esto coincida con la forma esperada por el modelo

        if model.get_learning_stage() == 'university':
            true_solution_tensor = tf.constant([[np.abs(true_solution)]], dtype=tf.float32)
        else:
            if hasattr(true_solution, 'real') and hasattr(true_solution, 'imag'):
                true_solution_tensor = tf.constant([[true_solution.real, true_solution.imag]], dtype=tf.float32)
            else:
                true_solution_tensor = tf.constant([[true_solution]], dtype=tf.float32)

        print(f"Forma de inputs antes de forward pass: {inputs.shape}")

        with tf.GradientTape() as tape:
            predicted = model(inputs, training=True)
            predicted = tf.reshape(predicted, true_solution_tensor.shape)  # Asegurar forma correcta
            print(f"Forma de predicted: {predicted.shape}")
            print(f"Forma de true_solution_tensor: {true_solution_tensor.shape}")
            loss = tf.reduce_mean(tf.square(tf.cast(true_solution_tensor, tf.float32) - tf.cast(predicted, tf.float32)))

        gradients = tape.gradient(loss, model.trainable_variables)

        if not any(grad is not None for grad in gradients):
            raise ValueError("No gradients were computed during the forward pass")

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return loss.numpy()
    except Exception as e:
        print(f"Error en reinforce_single: {str(e)}")
        raise

def parallel_reinforce_learning(model, problems, predictions, true_solutions, learning_rate=0.01):
    reinforce_data = [(model, problem, prediction, true_solution) 
                      for problem, prediction, true_solution in zip(problems, predictions, true_solutions)]

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        losses = pool.map(reinforce_single, reinforce_data)

    return losses