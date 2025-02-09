import unittest
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)


def create_test_suite():
    """Create a test suite combining both unit and integration tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Run integration tests
   # suite.addTests(loader.loadTestsFromName('test_image_analysis_int'))
    # Run unit tests
    suite.addTests(loader.loadTestsFromName('test_image_analysis_unit'))
    
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    test_suite = create_test_suite()
    runner.run(test_suite)
