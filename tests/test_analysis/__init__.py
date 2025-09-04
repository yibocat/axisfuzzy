#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AxisFuzzy Analysis Module Test Suite

This package contains comprehensive tests for the axisfuzzy.analysis module,
including component tests, contract tests, dependency tests, pipeline tests,
model tests, and extension examples.

Test Structure:
- conftest.py: Test configuration and shared fixtures
- test_components.py: Core component functionality tests
- test_contracts.py: Data contract system tests
- test_dependencies.py: Dependency injection system tests
- test_pipeline.py: Data processing pipeline tests
- test_model.py: Analysis model comprehensive tests
- test_extension_example.py: Framework extensibility demonstrations

Usage:
    # Run all analysis tests
    pytest tests/test_analysis/ -v
    
    # Run specific test files
    pytest tests/test_analysis/test_components.py -v
    pytest tests/test_analysis/test_model.py -v
    
    # Run with coverage
    pytest tests/test_analysis/ --cov=axisfuzzy.analysis --cov-report=html
"""

__version__ = "1.0.0"
__author__ = "AxisFuzzy Development Team"

# Import all test modules for easy access
try:
    from . import conftest
    from . import test_components
    from . import test_contracts
    from . import test_dependencies
    from . import test_pipeline
    from . import test_model
    from . import test_extension_example
except ImportError as e:
    # Handle import errors gracefully during development
    import warnings
    warnings.warn(f"Could not import test module: {e}", ImportWarning)

# Test configuration
TEST_CONFIG = {
    'timeout': 30,  # Default test timeout in seconds
    'slow_timeout': 120,  # Timeout for slow tests
    'random_seed': 42,  # For reproducible tests
    'coverage_threshold': 95,  # Minimum coverage percentage
    'parallel_workers': 4,  # Number of parallel test workers
}

# Test markers documentation
TEST_MARKERS = {
    'unit': 'Unit tests for individual components',
    'integration': 'Integration tests for component interactions',
    'slow': 'Tests that take longer than 5 seconds',
    'performance': 'Performance and benchmark tests',
    'regression': 'Regression tests for bug fixes',
    'component': 'Component-specific tests',
    'contract': 'Contract system tests',
    'dependency': 'Dependency injection tests',
    'pipeline': 'Pipeline system tests',
    'model': 'Model system tests',
    'extension': 'Framework extension tests'
}

# Test statistics (updated based on actual test results)
TEST_STATISTICS = {
    'total_files': 7,
    'total_test_cases': 123,  # Approximate total based on all test files
    'components_tested': {
        'ToolNormalization': 'Normalization component',
        'ToolStatistics': 'Statistics component', 
        'ToolSimpleAggregation': 'Aggregation component',
        'Pipeline': 'Data processing pipeline',
        'DependencyContainer': 'Dependency injection container',
        'Contract': 'Data contract system'
    },
    'models_tested': {
        'SimpleTestModel': 'Basic two-step model',
        'ThreeStepTestModel': 'Three-step linear model',
        'BranchingTestModel': 'Branching model architecture',
        'MultiStepTestModel': 'Multi-step processing model'
    },
    'coverage_areas': [
        'Component interfaces and functionality',
        'Data contract validation',
        'Dependency injection and management',
        'Pipeline construction and execution',
        'Model building and testing',
        'Framework extensibility',
        'Error handling and edge cases',
        'Performance and debugging features'
    ]
}

# Export test modules and utilities
__all__ = [
    'conftest',
    'test_components', 
    'test_contracts',
    'test_dependencies',
    'test_pipeline',
    'test_model',
    'test_extension_example',
    'TEST_CONFIG',
    'TEST_MARKERS',
    'TEST_STATISTICS'
]

# Test suite information
def get_test_info():
    """
    Get comprehensive information about the test suite.
    
    Returns
    -------
    dict
        Dictionary containing test suite information including
        configuration, statistics, and available test modules.
    """
    return {
        'version': __version__,
        'author': __author__,
        'config': TEST_CONFIG,
        'markers': TEST_MARKERS,
        'statistics': TEST_STATISTICS,
        'modules': __all__[:-3]  # Exclude config constants
    }

def run_all_tests():
    """
    Programmatically run all tests in the analysis module.
    
    This function provides a convenient way to run the entire
    test suite from Python code.
    
    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    import subprocess
    import sys
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            'tests/test_analysis/', '-v'
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
            
        return result.returncode
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1

if __name__ == '__main__':
    # Allow running tests directly from this module
    import sys
    sys.exit(run_all_tests())