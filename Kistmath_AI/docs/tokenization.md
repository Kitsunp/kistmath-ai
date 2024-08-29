# Kistmath_AI/utils/tokenization.py

This file contains functions for tokenizing mathematical problems.

## Functions

### tokenize_problem(problem, stage, vocab_size=VOCAB_SIZE, max_length=MAX_LENGTH)

- **Description**: Tokenizes a problem string into a fixed-length sequence of integers.
- **Parameters**:
  - `problem`: The problem string to tokenize.
  - `stage`: The learning stage.
  - `vocab_size`: The size of the vocabulary.
  - `max_length`: The maximum length of the tokenized sequence.

### tokenize_basic_problem(problem, vocab_size, max_length)

- **Description**: Tokenizes a basic problem into a fixed-length sequence of integers.
- **Parameters**:
  - `problem`: The problem string to tokenize.
  - `vocab_size`: The size of the vocabulary.
  - `max_length`: The maximum length of the tokenized sequence.

### tokenize_advanced_problem(problem, vocab_size, max_length)

- **Description**: Tokenizes an advanced problem into a fixed-length sequence of integers.
- **Parameters**:
  - `problem`: The problem string to tokenize.
  - `vocab_size`: The size of the vocabulary.
  - `max_length`: The maximum length of the tokenized sequence.

### tokenize_calculus_problem(problem, max_length)

- **Description**: Tokenizes a calculus problem into a fixed-length sequence of coefficients and exponents.
- **Parameters**:
  - `problem`: The problem string to tokenize.
  - `max_length`: The maximum length of the tokenized sequence.

### pad_tokens(tokens, max_length)

- **Description**: Pads or truncates a list of tokens to a fixed length.
- **Parameters**:
  - `tokens`: The list of tokens to pad or truncate.
  - `max_length`: The fixed length to pad or truncate to.

## Dependencies

- `numpy`
- `tensorflow`
- `config.settings`