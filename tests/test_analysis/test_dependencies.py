#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for AxisFuzzy analysis module dependencies and imports.

This module tests the analysis-specific dependency architecture and import mechanisms
for the axisfuzzy.analysis module. It focuses on testing the analysis module's
dependency checking functionality and integration with the core package.

Note: General dependency tests are located in tests/test_dependencies/
"""

import pytest
import sys
from typing import Dict, Any


class TestAnalysisDependencies:
    """Test analysis module dependencies and imports."""
    
    def test_analysis_import(self):
        """Test analysis module import functionality."""
        from axisfuzzy import analysis
        
        # Verify analysis module imports successfully
        assert analysis is not None
        
        # Verify dependency check function exists
        assert hasattr(analysis, 'check_analysis_dependencies')
        assert callable(analysis.check_analysis_dependencies)
    
    def test_dependency_check_function(self):
        """Test the dependency checking functionality."""
        from axisfuzzy.analysis import check_analysis_dependencies
        
        # Get dependency status
        deps = check_analysis_dependencies()
        
        # Should return a dictionary
        assert isinstance(deps, dict)
        
        # Should contain expected dependencies
        expected_deps = ['pandas', 'matplotlib', 'networkx', 'pydot', 'graphviz']
        for dep in expected_deps:
            assert dep in deps, f"Missing dependency check for {dep}"
            
            # Each dependency should have required fields
            dep_info = deps[dep]
            assert isinstance(dep_info, dict)
            assert 'installed' in dep_info
            assert 'version' in dep_info
            assert isinstance(dep_info['installed'], bool)
    
    def test_dependency_status_format(self):
        """Test that dependency status has correct format."""
        from axisfuzzy.analysis import check_analysis_dependencies
        
        deps = check_analysis_dependencies()
        
        for dep_name, dep_info in deps.items():
            # Verify structure
            assert isinstance(dep_name, str)
            assert isinstance(dep_info, dict)
            
            # Required fields
            assert 'installed' in dep_info
            assert 'version' in dep_info
            
            # Type checking
            assert isinstance(dep_info['installed'], bool)
            
            # If installed, version should be string or None
            if dep_info['installed']:
                assert dep_info['version'] is None or isinstance(dep_info['version'], str)
            else:
                # If not installed, version should be None
                assert dep_info['version'] is None


class TestOptionalDependencies:
    """Test optional dependency handling and graceful degradation."""
    
    def test_graceful_degradation(self):
        """Test that missing optional dependencies don't break core functionality."""
        # This test verifies that the analysis module can be imported
        # even if some optional dependencies are missing
        from axisfuzzy import analysis
        
        # Should be able to check dependencies even if some are missing
        deps = analysis.check_analysis_dependencies()
        assert isinstance(deps, dict)


class TestDependencyErrorHandling:
    """Test error handling for dependency-related issues."""
    
    def test_import_error_handling(self):
        """Test handling of import errors for missing dependencies."""
        from axisfuzzy.analysis import check_analysis_dependencies
        
        # This should not raise an exception even if some deps are missing
        deps = check_analysis_dependencies()
        assert isinstance(deps, dict)
        
        # Verify that missing dependencies are properly reported
        for dep_name, dep_info in deps.items():
            if not dep_info['installed']:
                assert dep_info['version'] is None
                # Should have error information if available
                # (This is optional based on implementation)
    
    def test_dependency_check_completeness(self):
        """Test that all expected dependencies are checked."""
        from axisfuzzy.analysis import check_analysis_dependencies
        
        deps = check_analysis_dependencies()
        expected_deps = ['pandas', 'matplotlib', 'networkx', 'pydot', 'graphviz']
        
        for dep in expected_deps:
            assert dep in deps, f"Missing dependency check for {dep}"
    
    def test_dependency_check_robustness(self):
        """Test that dependency checking is robust to various error conditions."""
        from axisfuzzy.analysis import check_analysis_dependencies
        
        # Should not raise exceptions
        try:
            deps = check_analysis_dependencies()
            assert isinstance(deps, dict)
        except Exception as e:
            pytest.fail(f"Dependency check raised unexpected exception: {e}")


class TestAnalysisModuleIntegration:
    """Test integration with the main test framework."""
    
    def test_analysis_module_import(self):
        """Test that analysis module can be imported and integrated properly."""
        # This test ensures the analysis module integrates well with the core package
        from axisfuzzy import analysis
        assert analysis is not None
        
        # Test that the module has expected attributes
        assert hasattr(analysis, 'check_analysis_dependencies')
        assert hasattr(analysis, 'Model')
        assert hasattr(analysis, 'app')


# Note: This test file focuses specifically on the analysis module's dependency
# checking functionality and integration. General dependency availability tests
# (e.g., testing if pandas, matplotlib, etc. are installed) are handled in
# tests/test_dependencies/ to avoid duplication and maintain clear separation
# of concerns.
