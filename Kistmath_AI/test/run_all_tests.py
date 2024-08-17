import unittest
import sys
from io import StringIO
import os

# Import all test modules
from test_external_memory import TestExternalMemory, TestFormulativeMemory, TestConceptualMemory, TestShortTermMemory, TestLongTermMemory, TestInferenceMemory
from test_kistmat_ai import TestKistmatAI
from test_symbolic_reasoning import TestSymbolicReasoning
from test_utils import TestUtils
from test_curriculum_learning_and_data_generation import TestCurriculumLearningAndDataGeneration
import unittest
import numpy as np
import tensorflow as tf
from Kistmath_AI.models.kistmat_ai import Kistmat_AI
from Kistmath_AI.utils.data_generation import generate_dataset, MathProblem
from Kistmath_AI.training.curriculum_learning import smooth_curriculum_learning
from Kistmath_AI.config.settings import READINESS_THRESHOLDS
import pytest
import tensorflow as tf
import numpy as np
from Kistmath_AI.models.external_memory import (
    ExternalMemory, FormulativeMemory, ConceptualMemory,
    ShortTermMemory, LongTermMemory, InferenceMemory
)
import unittest
import numpy as np
from Kistmath_AI.models.kistmat_ai import Kistmat_AI
from Kistmath_AI.config.settings import VOCAB_SIZE, MAX_LENGTH
import unittest
from Kistmath_AI.models.symbolic_reasoning import SymbolicReasoning
import sympy as sp
def run_tests_and_save_results():
    # Create a test suite
    suite = unittest.TestSuite()

    # Add all test cases to the suite
    suite.addTest(unittest.makeSuite(TestExternalMemory))
    suite.addTest(unittest.makeSuite(TestFormulativeMemory))
    suite.addTest(unittest.makeSuite(TestConceptualMemory))
    suite.addTest(unittest.makeSuite(TestShortTermMemory))
    suite.addTest(unittest.makeSuite(TestLongTermMemory))
    suite.addTest(unittest.makeSuite(TestInferenceMemory))
    suite.addTest(unittest.makeSuite(TestKistmatAI))
    suite.addTest(unittest.makeSuite(TestSymbolicReasoning))
    suite.addTest(unittest.makeSuite(TestUtils))
    suite.addTest(unittest.makeSuite(TestCurriculumLearningAndDataGeneration))

    # Redirect stdout to capture the output
    stdout = sys.stdout
    sys.stdout = StringIO()

    # Run the tests
    result = unittest.TextTestRunner(verbosity=2).run(suite)

    # Get the output
    output = sys.stdout.getvalue()

    # Restore stdout
    sys.stdout = stdout

    # Write results to file
    with open('test_results.txt', 'w') as f:
        f.write(output)
        f.write(f"\n\nRan {result.testsRun} tests\n")
        f.write(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}\n")
        f.write(f"Failures: {len(result.failures)}\n")
        f.write(f"Errors: {len(result.errors)}\n")

    print(f"Test results have been saved to {os.path.abspath('test_results.txt')}")

if __name__ == '__main__':
    run_tests_and_save_results()