import unittest
import importlib

class TestModelsInit(unittest.TestCase):

    def test_all_variable(self):
        """
        Test that the __all__ variable in models.__init__ is correctly defined.
        """
        models_module = importlib.import_module("viralStoryGenerator.models")
        self.assertIsNotNone(models_module.__all__)
        expected_all = [
            'StoryGenerationRequest',
            'JobResponse',
            'StoryGenerationResult',
            'ErrorResponse',
            'HealthResponse',
            'JobStatusResponse',
            'ServiceStatusDetail'
        ]
        self.assertListEqual(sorted(models_module.__all__), sorted(expected_all))

    def test_imports(self):
        """
        Test that all models listed in __all__ can be imported from viralStoryGenerator.models.
        """
        models_module = importlib.import_module("viralStoryGenerator.models")
        self.assertTrue(hasattr(models_module, "__all__"), "__all__ is not defined in models module.")
        self.assertIsInstance(models_module.__all__, list, "__all__ is not a list.")
        for model_name in models_module.__all__:
            attr = getattr(models_module, model_name)
            self.assertIsNotNone(attr, f"Model {model_name} could not be imported.")

if __name__ == '__main__':
    unittest.main()
