# Constants
VOCAB_SIZE = 1000
MAX_LENGTH = 10
MAX_TERMS = 5

# Stages for curriculum learning
STAGES = ['elementary1', 'elementary2', 'elementary3', 'junior_high1', 'junior_high2', 
          'high_school1', 'high_school2', 'high_school3', 'university']

# Readiness thresholds for each stage
READINESS_THRESHOLDS = {
    'elementary1': 0.95,
    'elementary2': 0.93,
    'elementary3': 0.91,
    'junior_high1': 0.89,
    'junior_high2': 0.87,
    'high_school1': 0.85,
    'high_school2': 0.83,
    'high_school3': 0.81,
    'university': 0.80
}

# Training settings
INITIAL_PROBLEMS = 4000
MAX_PROBLEMS = 5000
DIFFICULTY_INCREASE_RATE = 0.05