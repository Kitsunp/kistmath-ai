# Kistmath_AI/utils/evaluation.py

This file contains functions for evaluating the readiness of the model.

## Functions

### evaluate_readiness(model, problems, threshold)

- **Description**: Evaluates if the model is ready to advance to the next learning stage.
- **Parameters**:
  - `model`: The Kistmat_AI model to evaluate.
  - `problems`: A list of MathProblem instances to use for evaluation.
  - `threshold`: The R-squared threshold for considering the model ready.

#### Example
```python
from Kistmath_AI.utils.evaluation import evaluate_readiness
from Kistmath_AI.models.kistmat_ai import Kistmat_AI
from Kistmath_AI.utils.data_generation import generate_dataset

# Initialize the model
model = Kistmat_AI(input_shape=(100,), output_shape=(1,))

# Generate a dataset
problems = generate_dataset(num_problems=100, stage="basic", difficulty=1)

# Evaluate the model's readiness
is_ready = evaluate_readiness(model, problems, threshold=0.9)
print(f"Model readiness: {is_ready}")
```

## Dependencies

- `numpy`
- `utils.tokenization`