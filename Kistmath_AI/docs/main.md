# Kistmath_AI/main.py

This file contains the main function to run the Kistmat AI training process.

## Functions

### main()

- **Description**: Main function to run the Kistmat AI training process.
- **Details**: This function initializes the Kistmat AI model, configures TensorFlow settings, and starts the training process using smooth curriculum learning. It also handles real-time plotting of training progress and evaluates the model on test problems.

### safe_tokenize_problem(problem, learning_stage)

- **Description**: Safely tokenizes a problem string into a fixed-length sequence of integers.
- **Parameters**:
  - `problem`: The problem string to tokenize.
  - `learning_stage`: The current learning stage of the model.
- **Returns**: A numpy array representing the tokenized problem.

## Dependencies

- `os`
- `tensorflow`
- `numpy`
- `multiprocessing`
- `models.kistmat_ai`
- `models.external_memory`
- `training.curriculum_learning`
- `utils.data_generation`
- `utils.evaluation`
- `utils.tokenization`
- `training.parallel_training`
- `visualization.plotting`
- `config.settings`

## Detailed Description

### TensorFlow Configuration

The script configures TensorFlow to use all available CPU cores and to use memory more efficiently. It also detects if a GPU is available and configures TensorFlow to use it if possible.

### Model Initialization

The `Kistmat_AI` model is initialized with the specified input and output shapes. The model is then compiled with the Adam optimizer, mean squared error loss, and mean absolute error metric.

### Training Process

The training process is managed by the `smooth_curriculum_learning` function, which implements smooth curriculum learning for the Kistmat AI model. The training progress is plotted in real-time using a separate process.

### Evaluation

After training, the model is evaluated on a set of test problems. The mean squared error of the predictions is calculated and printed, along with a sample of predictions and their corresponding actual solutions.

### Reinforcement Learning

The script also performs reinforcement learning on a sample of test problems to further improve the model's performance.

### Error Handling

The script includes error handling to catch and print any exceptions that occur during the training and evaluation process.

## Example Usage

To run the Kistmat AI training process, simply execute the script:

```bash
python Kistmath_AI/main.py
```