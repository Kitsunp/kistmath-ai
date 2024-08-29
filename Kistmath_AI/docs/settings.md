# Kistmath_AI/config/settings.py

This file contains configuration settings for the Kistmat AI model.

## Constants

- `VOCAB_SIZE`: The size of the vocabulary.
- `MAX_LENGTH`: The maximum length of the tokenized sequence.
- `MAX_TERMS`: The maximum number of terms in a calculus problem.

## Stages for Curriculum Learning

- `STAGES`: A list of learning stages.

## Readiness Thresholds for Each Stage

- `READINESS_THRESHOLDS`: A dictionary of R-squared thresholds for each learning stage.

## Training Settings

- `INITIAL_PROBLEMS`: Initial number of problems per stage.
- `MAX_PROBLEMS`: Maximum number of problems per stage.
- `DIFFICULTY_INCREASE_RATE`: Rate at which difficulty increases.