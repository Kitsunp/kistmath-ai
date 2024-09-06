import unittest
import numpy as np
import tensorflow as tf
import time
from models.kistmat_ai import Kistmat_AI
from utils.data_generation import generate_dataset, MathProblem
from training.curriculum_learning import smooth_curriculum_learning
from config.settings import READINESS_THRESHOLDS

class TestCurriculumLearningAndDataGeneration(unittest.TestCase):
    
    def setUp(self):
        self.model = Kistmat_AI(input_shape=(100,), output_shape=1)
        self.stages = ['elementary1', 'elementary2', 'elementary3', 'junior_high1', 'junior_high2', 
                       'high_school1', 'high_school2', 'high_school3', 'university']
    
    def test_data_generation_consistency(self):
        """Test if data generation is consistent across multiple calls"""
        for stage in self.stages:
            data1 = generate_dataset(100, stage, 1.0)
            data2 = generate_dataset(100, stage, 1.0)
            self.assertEqual(len(data1), len(data2))
            self.assertEqual(type(data1[0]), type(data2[0]))
    
    def test_data_generation_difficulty_scaling(self):
        """Test if data difficulty scales appropriately"""
        for stage in self.stages:
            easy_data = generate_dataset(100, stage, 1.0)
            hard_data = generate_dataset(100, stage, 2.0)
            self.assertGreater(np.mean([p.difficulty for p in hard_data]),
                               np.mean([p.difficulty for p in easy_data]))
    
    def test_data_generation_stage_progression(self):
        """Test if data complexity increases with stage progression"""
        prev_complexity = 0
        for stage in self.stages:
            data = generate_dataset(100, stage, 1.0)
            complexity = np.mean([len(str(p.problem)) for p in data])
            self.assertGreater(complexity, prev_complexity)
            prev_complexity = complexity
    
    def test_data_generation_concept_coverage(self):
        """Test if generated data covers all expected concepts for each stage"""
        expected_concepts = {
            'elementary1': {'+', '-'},
            'elementary2': {'+', '-', '*'},
            'elementary3': {'+', '-', '*', '/'},
            'junior_high1': {'linear_equation', 'simple_inequalities'},
            'junior_high2': {'quadratic', 'roots', 'factorization'},
            'high_school1': {'logarithm', 'exponential'},
            'high_school2': {'trigonometry', 'sine', 'cosine', 'tangent'},
            'high_school3': {'limits', 'continuity'},
            'university': {'derivatives', 'integrals', 'differential_equations'}
        }
        
        for stage, concepts in expected_concepts.items():
            data = generate_dataset(1000, stage, 1.0)
            generated_concepts = set(p.concept for p in data)
            self.assertTrue(concepts.issubset(generated_concepts))
    
    def test_curriculum_learning_stage_progression(self):
        """Test if curriculum learning progresses through all stages"""
        history = smooth_curriculum_learning(self.model, self.stages, initial_problems=10, max_problems=20)
        stages_learned = [h['stage'] for h in history]
        self.assertEqual(stages_learned, self.stages)
    
    def test_curriculum_learning_difficulty_increase(self):
        """Test if problem difficulty increases within and across stages"""
        history = smooth_curriculum_learning(self.model, self.stages[:3], initial_problems=10, max_problems=20)
        for stage_history in history:
            difficulties = [h['difficulty'] for h in stage_history['fold_histories']]
            self.assertTrue(all(difficulties[i] <= difficulties[i+1] for i in range(len(difficulties)-1)))
    
    def test_curriculum_learning_model_improvement(self):
        """Test if model performance improves during curriculum learning"""
        history = smooth_curriculum_learning(self.model, self.stages[:3], initial_problems=10, max_problems=20)
        for stage_history in history:
            losses = [h['history']['loss'][-1] for h in stage_history['fold_histories']]
            self.assertTrue(all(losses[i] >= losses[i+1] for i in range(len(losses)-1)))
    
    def test_curriculum_learning_readiness_evaluation(self):
        """Test if readiness evaluation is working correctly"""
        stage = 'elementary1'
        problems = generate_dataset(100, stage, 1.0)
        
        # Train the model to perform well
        for _ in range(10):
            self.model.fit([p.problem for p in problems], [p.solution.real for p in problems], epochs=5)
        
        readiness = self.model.evaluate_readiness(problems, READINESS_THRESHOLDS[stage])
        self.assertTrue(readiness)
    
    def test_data_generation_input_shape_compatibility(self):
        """Test if generated data is compatible with model input shape"""
        for stage in self.stages:
            data = generate_dataset(100, stage, 1.0)
            problem = data[0].problem
            self.assertTrue(len(problem) <= self.model.input_shape[0])
    
    def test_data_generation_output_shape_compatibility(self):
        """Test if generated solutions are compatible with model output shape"""
        for stage in self.stages:
            data = generate_dataset(100, stage, 1.0)
            solution = data[0].solution
            self.assertEqual(solution.shape, (self.model.output_shape,))
    
    def test_curriculum_learning_memory_usage(self):
        """Test if curriculum learning manages memory efficiently"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        smooth_curriculum_learning(self.model, self.stages[:3], initial_problems=10, max_problems=20)
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / initial_memory
        
        self.assertLess(memory_increase, 0.5)  # Ensure memory usage doesn't increase by more than 50%
    
    def test_extreme_cases(self):
        """Test data generation and curriculum learning with extreme cases"""
        # Test with very large number of problems
        large_dataset = generate_dataset(10000, 'elementary1', 1.0)
        self.assertEqual(len(large_dataset), 10000)
        
        # Test with very high difficulty
        hard_dataset = generate_dataset(100, 'university', 5.0)
        self.assertTrue(all(p.difficulty >= 4.5 for p in hard_dataset))
        
        # Test curriculum learning with very short stages
        short_history = smooth_curriculum_learning(self.model, self.stages, initial_problems=1, max_problems=2)
        self.assertEqual(len(short_history), len(self.stages))
    
    def test_performance(self):
        """Test performance of data generation and curriculum learning"""
        start_time = time.time()
        generate_dataset(1000, 'high_school3', 2.0)
        end_time = time.time()
        self.assertLess(end_time - start_time, 5)  # Data generation should take less than 5 seconds
        
        start_time = time.time()
        smooth_curriculum_learning(self.model, self.stages[:2], initial_problems=10, max_problems=20)
        end_time = time.time()
        self.assertLess(end_time - start_time, 60)  # Curriculum learning for 2 stages should take less than 1 minute
    
    def test_result_consistency(self):
        """Test consistency of results across multiple runs"""
        results = []
        for _ in range(5):
            history = smooth_curriculum_learning(self.model, self.stages[:2], initial_problems=10, max_problems=20)
            results.append(history[-1]['fold_histories'][-1]['history']['loss'][-1])
        
        # Check if all results are within 10% of the mean
        mean_result = np.mean(results)
        self.assertTrue(all(abs(r - mean_result) / mean_result < 0.1 for r in results))
    
    def test_robustness(self):
        """Test robustness of the model to atypical or erroneous data"""
        # Generate some atypical data
        atypical_data = [
            MathProblem("What is x?", "x", 'elementary1', 1.0),  # Ambiguous problem
            MathProblem("2 + 2 = ?", 5, 'elementary1', 1.0),  # Incorrect solution
            MathProblem("", 0, 'elementary1', 1.0),  # Empty problem
            MathProblem("Very long problem " * 100, 42, 'elementary1', 1.0),  # Extremely long problem
        ]
        
        # Train the model on normal data
        normal_data = generate_dataset(100, 'elementary1', 1.0)
        self.model.fit([p.problem for p in normal_data], [p.solution.real for p in normal_data], epochs=5)
        
        # Test the model on atypical data
        for problem in atypical_data:
            try:
                self.model.predict([problem.problem])
            except Exception as e:
                self.fail(f"Model failed to handle atypical data: {str(e)}")

if __name__ == '__main__':
    unittest.main()