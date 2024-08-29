# Kistmath_AI/utils/evaluation.py

This file contains functions for evaluating the readiness of the model.

## Functions

### evaluate_readiness(model, problems, threshold)

- **Description**: Evaluates if the model is ready to advance to the next learning stage.
- **Parameters**:
  - `model`: The Kistmat_AI model to evaluate.
  - `problems`: A list of MathProblem instances to use for evaluation.
  - `threshold`: The R-squared threshold for considering the model ready.

## Dependencies

- `numpy`
- `utils.tokenization`