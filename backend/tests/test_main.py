"""Call all tests in a single test suite."""
import sys
import unittest
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def create_test_suite():
    """Create a test suite combining both unit and integration tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Run unit tests
    suite.addTests(loader.loadTestsFromName("test_image_analysis_unit"))

    # Run image analysis sanity tests
    suite.addTests(loader.loadTestsFromName("test_image_analysis_integration"))

    # Run API tests
    suite.addTests(loader.loadTestsFromName("test_api"))

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    test_suite = create_test_suite()
    runner.run(test_suite)
