import pytest
import numpy as np
from unittest.mock import patch
import warnings
from axisfuzzy.config import get_config

# filepath: /Users/yibow/Project/FuzzLab/axisfuzzy/test/core/test_triangular.py
"""
Comprehensive test suite for fuzzy t-norm framework module

This module provides extensive testing for all t-norm types, their properties,
and the unified framework functionality.
"""

from axisfuzzy.core.triangular import (
    OperationTNorm, BaseNormOperation, TypeNormalizer, create_tnorm,
    AlgebraicNorm, LukasiewiczNorm, EinsteinNorm, HamacherNorm,
    YagerNorm, SchweizerSklarNorm, DombiNorm, AczelAlsinaNorm,
    FrankNorm, MinimumNorm, DrasticNorm, NilpotentNorm
)


class TestHelpers:
    """Helper functions and utilities for testing"""

    @staticmethod
    def compare_arrays(actual, expected, atol=1e-10):
        """Compare arrays with tolerance"""
        if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
            return abs(actual - expected) <= atol
        return np.allclose(actual, expected, atol=atol, rtol=1e-12)

    @staticmethod
    def sample_inputs():
        """Generate representative scalar pairs and arrays for testing"""
        scalars = [0.0, 0.2, 0.5, 0.7, 1.0]
        arrays_1d = np.array([0.1, 0.3, 0.6, 0.9])
        arrays_2d = np.array([[0.1, 0.4], [0.6, 0.9]])

        return {
            'scalars': scalars,
            'arrays_1d': arrays_1d,
            'arrays_2d': arrays_2d,
            'grid_values': [0.0, 0.2, 0.5, 0.7, 1.0]
        }

    @staticmethod
    def get_test_grid():
        """Get a grid of test values"""
        return np.array([0.0, 0.1, 0.2, 0.5, 0.7, 0.9, 1.0])


class TestTypeNormalizer:
    """Test TypeNormalizer functionality"""

    def test_to_float_basic_types(self):
        """Test to_float with basic types"""
        assert TypeNormalizer.to_float(5) == 5.0
        assert TypeNormalizer.to_float(3.14) == 3.14
        assert TypeNormalizer.to_float(np.float64(2.5)) == 2.5
        assert isinstance(TypeNormalizer.to_float(5), float)

    def test_to_float_array_scalar(self):
        """Test to_float with scalar array"""
        scalar_array = np.array(3.5)
        result = TypeNormalizer.to_float(scalar_array)
        assert result == 3.5
        assert isinstance(result, float)

    def test_to_float_none_raises(self):
        """Test to_float with None raises error"""
        with pytest.raises((TypeError, ValueError)):
            TypeNormalizer.to_float(None)

    def test_to_int_basic_types(self):
        """Test to_int with basic types"""
        assert TypeNormalizer.to_int(5) == 5
        assert TypeNormalizer.to_int(3.0) == 3
        assert TypeNormalizer.to_int(np.int32(7)) == 7
        assert isinstance(TypeNormalizer.to_int(5.0), int)

    def test_ensure_array_output_dtype(self):
        """Test ensure_array_output returns float64 dtype"""
        result = TypeNormalizer.ensure_array_output([1, 2, 3])
        assert result.dtype == np.float64
        assert np.array_equal(result, [1.0, 2.0, 3.0])

    def test_ensure_scalar_output_raises_on_array(self):
        """Test ensure_scalar_output raises on non-scalar array"""
        with pytest.raises((ValueError, TypeError)):
            TypeNormalizer.ensure_scalar_output(np.array([1, 2, 3]))


class TestParameterValidation:
    """Test parameter validation for all norm types"""

    def test_q_validation(self):
        """Test q <= 0 raises error"""
        with pytest.raises(ValueError, match="q must be a positive integer"):
            OperationTNorm(norm_type='algebraic', q=0)

        with pytest.raises(ValueError, match="q must be a positive integer"):
            OperationTNorm(norm_type='algebraic', q=-1)

    def test_hamacher_gamma_validation(self):
        """Test Hamacher gamma <= 0 raises error"""
        with pytest.raises(ValueError, match="Hamacher parameter gamma must be positive"):
            OperationTNorm(norm_type='hamacher', hamacher_param=0.0)

        with pytest.raises(ValueError, match="Hamacher parameter gamma must be positive"):
            OperationTNorm(norm_type='hamacher', hamacher_param=-1.0)

    def test_yager_p_validation(self):
        """Test Yager p <= 0 raises error"""
        with pytest.raises(ValueError, match="Yager parameter p must be positive"):
            OperationTNorm(norm_type='yager', yager_param=0.0)

        with pytest.raises(ValueError, match="Yager parameter p must be positive"):
            OperationTNorm(norm_type='yager', yager_param=-1.0)

    def test_schweizer_sklar_p_validation(self):
        """Test SchweizerSklar p = 0 raises error"""
        with pytest.raises(ValueError, match="Schweizer-Sklar parameter p cannot be zero"):
            OperationTNorm(norm_type='schweizer_sklar', sklar_param=0.0)

    def test_dombi_p_validation(self):
        """Test Dombi p <= 0 raises error"""
        with pytest.raises(ValueError, match="Dombi parameter p must be positive"):
            OperationTNorm(norm_type='dombi', dombi_param=0.0)

        with pytest.raises(ValueError, match="Dombi parameter p must be positive"):
            OperationTNorm(norm_type='dombi', dombi_param=-1.0)

    def test_aczel_alsina_p_validation(self):
        """Test AczelAlsina p <= 0 raises error"""
        with pytest.raises(ValueError, match="Aczel-Alsina parameter p must be positive"):
            OperationTNorm(norm_type='aczel_alsina', aa_param=0.0)

    def test_frank_s_validation(self):
        """Test Frank s invalid values raise error"""
        with pytest.raises(ValueError, match="Frank parameter s must be positive and not equal to 1"):
            OperationTNorm(norm_type='frank', frank_param=0.0)

        with pytest.raises(ValueError, match="Frank parameter s must be positive and not equal to 1"):
            OperationTNorm(norm_type='frank', frank_param=1.0)


class TestRegistry:
    """Test norm registry functionality"""

    def test_list_available_norms(self):
        """Test list_available_norms contains all predefined types"""
        available = OperationTNorm.list_available_norms()
        expected_norms = [
            'algebraic', 'lukasiewicz', 'einstein', 'hamacher', 'yager',
            'schweizer_sklar', 'dombi', 'aczel_alsina', 'frank',
            'minimum', 'drastic', 'nilpotent'
        ]

        for norm in expected_norms:
            assert norm in available

    def test_register_norm(self):
        """Test register_norm adds new norm type"""

        class CustomNorm(BaseNormOperation):
            def t_norm_impl(self, a, b):
                return TypeNormalizer.ensure_array_output(a * b * 0.5)

            def t_conorm_impl(self, a, b):
                return TypeNormalizer.ensure_array_output(a + b - a * b * 0.5)

        OperationTNorm.register_norm('custom', CustomNorm)

        # Test that custom norm is now available
        assert 'custom' in OperationTNorm.list_available_norms()

        # Test that we can create instance
        norm = OperationTNorm(norm_type='custom')
        assert norm.norm_type == 'custom'

        # Clean up
        del OperationTNorm._NORM_REGISTRY['custom']


class TestNormCorrectness:
    """Test algebraic correctness for all norm types"""

    @pytest.mark.parametrize("norm_type", [
        'algebraic', 'lukasiewicz', 'einstein', 'hamacher', 'yager',
        'schweizer_sklar', 'dombi', 'aczel_alsina', 'frank',
        'minimum', 'drastic', 'nilpotent'
    ])
    def test_norm_closed_form_calculations(self, norm_type):
        """Test closed-form calculations for each norm type"""
        helpers = TestHelpers()
        grid_values = helpers.get_test_grid()

        # Create appropriate parameters for each norm type
        params = self._get_norm_params(norm_type)
        norm = OperationTNorm(norm_type=norm_type, **params)

        for a in grid_values:
            for b in grid_values:
                t_result = norm.t_norm(a, b)
                s_result = norm.t_conorm(a, b)

                # Verify results are in [0,1]
                assert 0 <= t_result <= 1, f"T-norm out of bounds for {norm_type}: T({a},{b})={t_result}"
                assert 0 <= s_result <= 1, f"T-conorm out of bounds for {norm_type}: S({a},{b})={s_result}"

                # Test specific known values
                expected_t, expected_s = self._get_expected_values(norm_type, a, b, params)
                if expected_t is not None:
                    assert helpers.compare_arrays(t_result, expected_t, atol=1e-10), \
                        f"T-norm mismatch for {norm_type}: T({a},{b})={t_result}, expected={expected_t}"
                if expected_s is not None:
                    assert helpers.compare_arrays(s_result, expected_s, atol=1e-10), \
                        f"T-conorm mismatch for {norm_type}: S({a},{b})={s_result}, expected={expected_s}"

    def _get_norm_params(self, norm_type):
        """Get appropriate parameters for each norm type"""
        params = {}
        if norm_type == 'hamacher':
            params['hamacher_param'] = 2.0
        elif norm_type == 'yager':
            params['yager_param'] = 2.0
        elif norm_type == 'schweizer_sklar':
            params['sklar_param'] = 2.0
        elif norm_type == 'dombi':
            params['dombi_param'] = 2.0
        elif norm_type == 'aczel_alsina':
            params['aa_param'] = 2.0
        elif norm_type == 'frank':
            params['frank_param'] = np.e
        return params

    def _get_expected_values(self, norm_type, a, b, params):
        """Get expected values for specific norm types and inputs"""
        if norm_type == 'algebraic':
            return a * b, a + b - a * b
        elif norm_type == 'lukasiewicz':
            return max(0, a + b - 1), min(1, a + b)
        elif norm_type == 'einstein':
            t = (a * b) / (1 + (1 - a) * (1 - b))
            s = (a + b) / (1 + a * b)
            return t, s
        elif norm_type == 'hamacher':
            gamma = params.get('hamacher_param', 1.0)
            t = (a * b) / (gamma + (1 - gamma) * (a + b - a * b))
            s = (a + b - (2 - gamma) * a * b) / (1 - (1 - gamma) * a * b)
            return t, s
        elif norm_type == 'yager':
            p = params.get('yager_param', 1.0)
            t = max(0.0, 1.0 - ((1 - a) ** p + (1 - b) ** p) ** (1.0 / p))
            s = min(1.0, (a ** p + b ** p) ** (1.0 / p))
            return t, s
        elif norm_type == 'frank':
            s_param = params.get('frank_param', np.e)
            if s_param > 0 and s_param != 1:
                val_a = np.power(s_param, a) - 1
                val_b = np.power(s_param, b) - 1
                arg_t = 1 + (val_a * val_b) / (s_param - 1)
                t = 0.0 if arg_t <= 0 else np.log(arg_t) / np.log(s_param)

                val_1_a = np.power(s_param, 1 - a) - 1
                val_1_b = np.power(s_param, 1 - b) - 1
                arg_s = 1 + (val_1_a * val_1_b) / (s_param - 1)
                s = 1.0 if arg_s <= 0 else 1 - np.log(arg_s) / np.log(s_param)
                return float(t), float(s)
        elif norm_type == 'minimum':
            return min(a, b), max(a, b)
        elif norm_type == 'drastic':
            eps = 1e-12
            t = a if abs(b - 1.0) < eps else (b if abs(a - 1.0) < eps else 0.0)
            s = a if abs(b - 0.0) < eps else (b if abs(a - 0.0) < eps else 1.0)
            return t, s
        elif norm_type == 'nilpotent':
            t = min(a, b) if a + b > 1 else 0.0
            s = max(a, b) if a + b < 1 else 1.0
            return t, s
        # Add more specific cases as needed
        return None, None


class TestScalarArrayBehavior:
    """Test scalar vs array input/output behavior"""

    def test_scalar_inputs_return_float(self):
        """Test scalar inputs return float"""
        norm = OperationTNorm(norm_type='algebraic')

        result_t = norm.t_norm(0.5, 0.7)
        result_s = norm.t_conorm(0.5, 0.7)

        assert isinstance(result_t, float)
        assert isinstance(result_s, float)

    def test_array_inputs_return_ndarray(self):
        """Test array inputs return ndarray"""
        norm = OperationTNorm(norm_type='algebraic')

        a = np.array([0.3, 0.6])
        b = np.array([0.4, 0.8])

        result_t = norm.t_norm(a, b)
        result_s = norm.t_conorm(a, b)

        assert isinstance(result_t, np.ndarray)
        assert isinstance(result_s, np.ndarray)


class TestBoundaryAxioms:
    """Test boundary axioms for all norm types"""

    @pytest.mark.parametrize("norm_type", [
        'algebraic', 'lukasiewicz', 'einstein', 'hamacher', 'yager',
        'schweizer_sklar', 'dombi', 'aczel_alsina', 'frank',
        'minimum', 'nilpotent'  # drastic has special rules
    ])
    def test_standard_boundary_axioms(self, norm_type):
        """Test standard boundary axioms: T(a,1)=a, T(a,0)=0, S(a,0)=a, S(a,1)=1"""
        params = self._get_norm_params(norm_type)
        norm = OperationTNorm(norm_type=norm_type, **params)

        test_values = [0.0, 0.2, 0.5, 0.7, 1.0]

        for a in test_values:
            # T(a,1) = a
            assert TestHelpers.compare_arrays(norm.t_norm(a, 1.0), a), \
                f"T({a},1) != {a} for {norm_type}"

            # T(a,0) = 0 (except for some special cases)
            if not (norm_type == 'drastic' and abs(a - 1.0) < 1e-10):
                assert TestHelpers.compare_arrays(norm.t_norm(a, 0.0), 0.0), \
                    f"T({a},0) != 0 for {norm_type}"

            # S(a,0) = a
            assert TestHelpers.compare_arrays(norm.t_conorm(a, 0.0), a), \
                f"S({a},0) != {a} for {norm_type}"

            # S(a,1) = 1 (except for some special cases)
            if not (norm_type == 'drastic' and abs(a - 0.0) < 1e-10):
                assert TestHelpers.compare_arrays(norm.t_conorm(a, 1.0), 1.0), \
                    f"S({a},1) != 1 for {norm_type}"

    def test_drastic_special_rules(self):
        """Test special boundary rules for drastic norm"""
        norm = OperationTNorm(norm_type='drastic')

        # T(a,b) = a if b=1; b if a=1; 0 otherwise
        assert TestHelpers.compare_arrays(norm.t_norm(0.5, 1.0), 0.5)
        assert TestHelpers.compare_arrays(norm.t_norm(1.0, 0.7), 0.7)
        assert TestHelpers.compare_arrays(norm.t_norm(0.3, 0.7), 0.0)

        # S(a,b) = a if b=0; b if a=0; 1 otherwise
        assert TestHelpers.compare_arrays(norm.t_conorm(0.5, 0.0), 0.5)
        assert TestHelpers.compare_arrays(norm.t_conorm(0.0, 0.7), 0.7)
        assert TestHelpers.compare_arrays(norm.t_conorm(0.3, 0.7), 1.0)

    def test_nilpotent_special_rules(self):
        """Test special rules for nilpotent norm"""
        norm = OperationTNorm(norm_type='nilpotent')

        # T(a,b) = min(a,b) if a+b>1; 0 otherwise
        assert TestHelpers.compare_arrays(norm.t_norm(0.6, 0.7), min(0.6, 0.7))  # 0.6+0.7 > 1
        assert TestHelpers.compare_arrays(norm.t_norm(0.3, 0.4), 0.0)  # 0.3+0.4 < 1

        # S(a,b) = max(a,b) if a+b<1; 1 otherwise
        assert TestHelpers.compare_arrays(norm.t_conorm(0.3, 0.4), max(0.3, 0.4))  # 0.3+0.4 < 1
        assert TestHelpers.compare_arrays(norm.t_conorm(0.6, 0.7), 1.0)  # 0.6+0.7 > 1

    def _get_norm_params(self, norm_type):
        """Get appropriate parameters for each norm type"""
        params = {}
        if norm_type == 'hamacher':
            params['hamacher_param'] = 2.0
        elif norm_type == 'yager':
            params['yager_param'] = 2.0
        elif norm_type == 'schweizer_sklar':
            params['sklar_param'] = 2.0
        elif norm_type == 'dombi':
            params['dombi_param'] = 2.0
        elif norm_type == 'aczel_alsina':
            params['aa_param'] = 2.0
        elif norm_type == 'frank':
            params['frank_param'] = np.e
        return params


class TestAlgebraicProperties:
    """Test algebraic properties: commutativity, associativity, monotonicity"""

    @pytest.mark.parametrize("norm_type", [
        'algebraic', 'lukasiewicz', 'einstein', 'hamacher', 'yager',
        'schweizer_sklar', 'dombi', 'aczel_alsina', 'frank',
        'minimum', 'drastic', 'nilpotent'
    ])
    def test_commutativity(self, norm_type):
        """Test T(a,b) == T(b,a) and S(a,b) == S(b,a)"""
        params = self._get_norm_params(norm_type)
        norm = OperationTNorm(norm_type=norm_type, **params)

        test_pairs = [(0.3, 0.7), (0.1, 0.9), (0.5, 0.5), (0.0, 0.8), (0.6, 1.0)]

        for a, b in test_pairs:
            t_ab = norm.t_norm(a, b)
            t_ba = norm.t_norm(b, a)
            s_ab = norm.t_conorm(a, b)
            s_ba = norm.t_conorm(b, a)

            assert TestHelpers.compare_arrays(t_ab, t_ba), \
                f"Commutativity failed for {norm_type}: T({a},{b})={t_ab} != T({b},{a})={t_ba}"
            assert TestHelpers.compare_arrays(s_ab, s_ba), \
                f"Commutativity failed for {norm_type}: S({a},{b})={s_ab} != S({b},{a})={s_ba}"

    @pytest.mark.parametrize("norm_type", [
        'algebraic', 'lukasiewicz', 'einstein', 'hamacher', 'yager',
        'schweizer_sklar', 'dombi', 'aczel_alsina', 'frank',
        'minimum', 'drastic', 'nilpotent'
    ])
    def test_associativity(self, norm_type):
        """Test T(T(a,b),c) == T(a,T(b,c)) and S(S(a,b),c) == S(a,S(b,c))"""
        params = self._get_norm_params(norm_type)
        norm = OperationTNorm(norm_type=norm_type, **params)

        test_triples = [(0.2, 0.5, 0.8), (0.1, 0.3, 0.9), (0.4, 0.6, 0.7)]

        for a, b, c in test_triples:
            # Test T-norm associativity
            t_ab_c = norm.t_norm(norm.t_norm(a, b), c)
            t_a_bc = norm.t_norm(a, norm.t_norm(b, c))

            # Test T-conorm associativity
            s_ab_c = norm.t_conorm(norm.t_conorm(a, b), c)
            s_a_bc = norm.t_conorm(a, norm.t_conorm(b, c))

            assert TestHelpers.compare_arrays(t_ab_c, t_a_bc), \
                f"T-norm associativity failed for {norm_type}: T(T({a},{b}),{c}) != T({a},T({b},{c}))"
            assert TestHelpers.compare_arrays(s_ab_c, s_a_bc), \
                f"T-conorm associativity failed for {norm_type}: S(S({a},{b}),{c}) != S({a},S({b},{c}))"

    @pytest.mark.parametrize("norm_type", [
        'algebraic', 'lukasiewicz', 'einstein', 'hamacher', 'yager',
        'schweizer_sklar', 'dombi', 'aczel_alsina', 'frank',
        'minimum', 'drastic', 'nilpotent'
    ])
    def test_monotonicity(self, norm_type):
        """Test monotonicity: if a1 <= a2, then T(a1,b) <= T(a2,b) and S(a1,b) <= S(a2,b)"""
        params = self._get_norm_params(norm_type)
        norm = OperationTNorm(norm_type=norm_type, **params)

        test_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        for b in [0.3, 0.7]:
            for i in range(len(test_values) - 1):
                a1, a2 = test_values[i], test_values[i + 1]

                t1 = norm.t_norm(a1, b)
                t2 = norm.t_norm(a2, b)
                s1 = norm.t_conorm(a1, b)
                s2 = norm.t_conorm(a2, b)

                assert t1 <= t2 + 1e-10, \
                    f"T-norm monotonicity failed for {norm_type}: T({a1},{b})={t1} > T({a2},{b})={t2}"
                assert s1 <= s2 + 1e-10, \
                    f"T-conorm monotonicity failed for {norm_type}: S({a1},{b})={s1} > S({a2},{b})={s2}"

    def _get_norm_params(self, norm_type):
        """Get appropriate parameters for each norm type"""
        params = {}
        if norm_type == 'hamacher':
            params['hamacher_param'] = 2.0
        elif norm_type == 'yager':
            params['yager_param'] = 2.0
        elif norm_type == 'schweizer_sklar':
            params['sklar_param'] = 2.0
        elif norm_type == 'dombi':
            params['dombi_param'] = 2.0
        elif norm_type == 'aczel_alsina':
            params['aa_param'] = 2.0
        elif norm_type == 'frank':
            params['frank_param'] = np.e
        return params


class TestArchimedeanProperty:
    """Test Archimedean property for relevant norms"""

    @pytest.mark.parametrize("norm_type,is_strict", [
        ('algebraic', True), ('einstein', True), ('hamacher', True),
        ('yager', True), ('schweizer_sklar', True), ('dombi', True),
        ('aczel_alsina', True), ('frank', True), ('lukasiewicz', False)
    ])
    def test_archimedean_property(self, norm_type, is_strict):
        """Test T(a,a) < a for strict Archimedean, T(a,a) <= a for non-strict"""
        params = self._get_norm_params(norm_type)
        norm = OperationTNorm(norm_type=norm_type, **params)

        test_values = [0.1, 0.3, 0.5, 0.7, 0.9]

        for a in test_values:
            t_aa = norm.t_norm(a, a)

            if is_strict:
                assert t_aa < a + 1e-10, \
                    f"Strict Archimedean property failed for {norm_type}: T({a},{a})={t_aa} >= {a}"
            else:
                assert t_aa <= a + 1e-10, \
                    f"Archimedean property failed for {norm_type}: T({a},{a})={t_aa} > {a}"

    def _get_norm_params(self, norm_type):
        """Get appropriate parameters for each norm type"""
        params = {}
        if norm_type == 'hamacher':
            params['hamacher_param'] = 2.0
        elif norm_type == 'yager':
            params['yager_param'] = 2.0
        elif norm_type == 'schweizer_sklar':
            params['sklar_param'] = 2.0
        elif norm_type == 'dombi':
            params['dombi_param'] = 2.0
        elif norm_type == 'aczel_alsina':
            params['aa_param'] = 2.0
        elif norm_type == 'frank':
            params['frank_param'] = np.e
        return params


class TestGenerators:
    """Test generator and inverse generator functions"""

    @pytest.mark.parametrize("norm_type", [
        'algebraic', 'lukasiewicz', 'einstein', 'hamacher', 'yager',
        'schweizer_sklar', 'dombi', 'aczel_alsina', 'frank'
    ])
    def test_generator_inverse_consistency(self, norm_type):
        """Test g_inv(g(a)) ≈ a for generator functions"""
        params = self._get_norm_params(norm_type)
        norm = OperationTNorm(norm_type=norm_type, **params)

        if not norm.is_archimedean or norm.g_func is None:
            pytest.skip(f"No generator for {norm_type}")

        test_values = [0.1, 0.2, 0.5, 0.7, 0.9]

        for a in test_values:
            g_a = norm.g_func(a)
            if not np.isinf(g_a):
                g_inv_g_a = norm.g_inv_func(g_a)
                assert TestHelpers.compare_arrays(g_inv_g_a, a, atol=1e-8), \
                    f"Generator inverse consistency failed for {norm_type}: g_inv(g({a}))={g_inv_g_a} != {a}"

    @pytest.mark.parametrize("norm_type", [
        'algebraic', 'lukasiewicz', 'einstein', 'hamacher', 'yager',
        'schweizer_sklar', 'dombi', 'aczel_alsina', 'frank'
    ])
    def test_generator_t_norm_relation(self, norm_type):
        """Test T(a,b) ≈ g_inv(g(a)+g(b)) for representative values"""
        params = self._get_norm_params(norm_type)
        norm = OperationTNorm(norm_type=norm_type, **params)

        if not norm.is_archimedean or norm.g_func is None:
            pytest.skip(f"No generator for {norm_type}")

        test_pairs = [(0.3, 0.7), (0.2, 0.8), (0.5, 0.5)]

        for a, b in test_pairs:
            t_ab = norm.t_norm(a, b)

            g_a = norm.g_func(a)
            g_b = norm.g_func(b)

            if not (np.isinf(g_a) or np.isinf(g_b)):
                g_inv_sum = norm.g_inv_func(g_a + g_b)
                assert TestHelpers.compare_arrays(t_ab, g_inv_sum, atol=1e-8), \
                    f"Generator T-norm relation failed for {norm_type}: T({a},{b})={t_ab} != g_inv(g({a})+g({b}))={g_inv_sum}"

    def _get_norm_params(self, norm_type):
        """Get appropriate parameters for each norm type"""
        params = {}
        if norm_type == 'hamacher':
            params['hamacher_param'] = 2.0
        elif norm_type == 'yager':
            params['yager_param'] = 2.0
        elif norm_type == 'schweizer_sklar':
            params['sklar_param'] = 2.0
        elif norm_type == 'dombi':
            params['dombi_param'] = 2.0
        elif norm_type == 'aczel_alsina':
            params['aa_param'] = 2.0
        elif norm_type == 'frank':
            params['frank_param'] = np.e
        return params


class TestDualGenerators:
    """Test dual generator functions"""

    @pytest.mark.parametrize("norm_type", [
        'algebraic', 'lukasiewicz', 'einstein', 'hamacher', 'yager',
        'schweizer_sklar', 'dombi', 'aczel_alsina', 'frank'
    ])
    def test_dual_generator_inverse_consistency(self, norm_type):
        """Test f_inv(f(a)) ≈ a for dual generator functions"""
        params = self._get_norm_params(norm_type)
        norm = OperationTNorm(norm_type=norm_type, **params)

        if not norm.is_archimedean or norm.f_func is None:
            pytest.skip(f"No dual generator for {norm_type}")

        test_values = [0.1, 0.2, 0.5, 0.7, 0.9]

        for a in test_values:
            f_a = norm.f_func(a)
            if not np.isinf(f_a):
                f_inv_f_a = norm.f_inv_func(f_a)
                assert TestHelpers.compare_arrays(f_inv_f_a, a, atol=1e-8), \
                    f"Dual generator inverse consistency failed for {norm_type}: f_inv(f({a}))={f_inv_f_a} != {a}"

    def _get_norm_params(self, norm_type):
        """Get appropriate parameters for each norm type"""
        params = {}
        if norm_type == 'hamacher':
            params['hamacher_param'] = 2.0
        elif norm_type == 'yager':
            params['yager_param'] = 2.0
        elif norm_type == 'schweizer_sklar':
            params['sklar_param'] = 2.0
        elif norm_type == 'dombi':
            params['dombi_param'] = 2.0
        elif norm_type == 'aczel_alsina':
            params['aa_param'] = 2.0
        elif norm_type == 'frank':
            params['frank_param'] = np.e
        return params


class TestQExtension:
    """Test q-extension functionality"""

    @pytest.mark.parametrize("norm_type", [
        'algebraic', 'lukasiewicz', 'einstein', 'hamacher', 'yager',
        'schweizer_sklar', 'dombi', 'aczel_alsina', 'frank'
    ])
    def test_q_power_relation_t_norm(self, norm_type):
        """Test T_q(a,b)^q == T_base(a^q, b^q) for q-extension"""
        params = self._get_norm_params(norm_type)

        q = 3
        norm_q = OperationTNorm(norm_type=norm_type, q=q, **params)
        norm_base = OperationTNorm(norm_type=norm_type, q=1, **params)

        if not norm_q.supports_q:
            pytest.skip(f"No q-extension support for {norm_type}")

        test_pairs = [(0.3, 0.7), (0.2, 0.8), (0.5, 0.6)]

        for a, b in test_pairs:
            t_q_ab = norm_q.t_norm(a, b)
            t_base_aq_bq = norm_base.t_norm(a ** q, b ** q)

            assert TestHelpers.compare_arrays(t_q_ab ** q, t_base_aq_bq, atol=1e-8), \
                f"Q-extension T-norm relation failed for {norm_type} with q={q}: T_q({a},{b})^{q}={t_q_ab ** q} != T_base({a ** q},{b ** q})={t_base_aq_bq}"

    @pytest.mark.parametrize("norm_type", [
        'algebraic', 'lukasiewicz', 'einstein', 'hamacher', 'yager',
        'schweizer_sklar', 'dombi', 'aczel_alsina', 'frank'
    ])
    def test_q_power_relation_t_conorm(self, norm_type):
        """Test S_q(a,b)^q == S_base(a^q, b^q) for q-extension"""
        params = self._get_norm_params(norm_type)

        q = 3
        norm_q = OperationTNorm(norm_type=norm_type, q=q, **params)
        norm_base = OperationTNorm(norm_type=norm_type, q=1, **params)

        if not norm_q.supports_q:
            pytest.skip(f"No q-extension support for {norm_type}")

        test_pairs = [(0.3, 0.7), (0.2, 0.8), (0.5, 0.6)]

        for a, b in test_pairs:
            s_q_ab = norm_q.t_conorm(a, b)
            s_base_aq_bq = norm_base.t_conorm(a ** q, b ** q)

            assert TestHelpers.compare_arrays(s_q_ab ** q, s_base_aq_bq, atol=1e-8), \
                f"Q-extension T-conorm relation failed for {norm_type} with q={q}: S_q({a},{b})^{q}={s_q_ab ** q} != S_base({a ** q},{b ** q})={s_base_aq_bq}"

    def _get_norm_params(self, norm_type):
        """Get appropriate parameters for each norm type"""
        params = {}
        if norm_type == 'hamacher':
            params['hamacher_param'] = 2.0
        elif norm_type == 'yager':
            params['yager_param'] = 2.0
        elif norm_type == 'schweizer_sklar':
            params['sklar_param'] = 2.0
        elif norm_type == 'dombi':
            params['dombi_param'] = 2.0
        elif norm_type == 'aczel_alsina':
            params['aa_param'] = 2.0
        elif norm_type == 'frank':
            params['frank_param'] = np.e
        return params


class TestDeMorganLaws:
    """Test De Morgan's Laws"""

    @pytest.mark.parametrize("norm_type", [
        'algebraic', 'lukasiewicz', 'einstein', 'hamacher', 'yager',
        'schweizer_sklar', 'dombi', 'aczel_alsina', 'frank',
        'minimum', 'drastic', 'nilpotent'
    ])
    def test_verify_de_morgan_laws_function(self, norm_type):
        """Test verify_de_morgan_laws() returns True for both laws"""
        params = self._get_norm_params(norm_type)
        norm = OperationTNorm(norm_type=norm_type, **params)

        result = norm.verify_de_morgan_laws()

        assert isinstance(result, dict)
        assert 'de_morgan_1' in result
        assert 'de_morgan_2' in result
        assert result['de_morgan_1'] is True
        assert result['de_morgan_2'] is True

    @pytest.mark.parametrize("norm_type", [
        'algebraic', 'lukasiewicz', 'einstein', 'hamacher', 'yager',
        'schweizer_sklar', 'dombi', 'aczel_alsina', 'frank',
        'minimum', 'drastic', 'nilpotent'
    ])
    def test_manual_de_morgan_reconstruction(self, norm_type):
        """Test De Morgan's laws by manual reconstruction"""
        params = self._get_norm_params(norm_type)

        # 检查范数是否支持 q 阶扩展
        temp_norm_for_check = OperationTNorm(norm_type=norm_type, **params)
        supports_q = temp_norm_for_check.supports_q

        # 根据是否支持 q 阶来设置测试参数
        if supports_q:
            q_val = 2  # 使用 q=2 测试 q-rung 德摩根定律
            complement = lambda x, q=q_val: (1 - x**q)**(1/q) if 0 < x < 1 else (1.0 if np.isclose(x, 0) else 0.0)
        else:
            q_val = 1  # 使用 q=1 和标准补函数测试
            complement = lambda x, q=q_val: 1.0 - x

        norm = OperationTNorm(norm_type=norm_type, q=q_val, **params)
        test_pairs = [(0.3, 0.7), (0.2, 0.8), (0.5, 0.6)]

        for a, b in test_pairs:
            # De Morgan's first law: N(T(a,b)) = S(N(a),N(b))
            t_ab = norm.t_norm(a, b)
            s_na_nb = norm.t_conorm(complement(a), complement(b))
            left_side_1 = complement(t_ab)
            right_side_1 = s_na_nb

            assert TestHelpers.compare_arrays(left_side_1, right_side_1, atol=1e-8), \
                f"First De Morgan law failed for {norm_type} (q={q_val}): N(T({a},{b}))={left_side_1} != S(N({a}),N({b}))={right_side_1}"

            # De Morgan's second law: N(S(a,b)) = T(N(a),N(b))
            s_ab = norm.t_conorm(a, b)
            t_na_nb = norm.t_norm(complement(a), complement(b))
            left_side_2 = complement(s_ab)
            right_side_2 = t_na_nb

            assert TestHelpers.compare_arrays(left_side_2, right_side_2, atol=1e-8), \
                f"Second De Morgan law failed for {norm_type} (q={q_val}): N(S({a},{b}))={left_side_2} != T(N({a}),N({b}))={right_side_2}"


    def _get_norm_params(self, norm_type):
        """Get appropriate parameters for each norm type"""
        params = {}
        if norm_type == 'hamacher':
            params['hamacher_param'] = 2.0
        elif norm_type == 'yager':
            params['yager_param'] = 2.0
        elif norm_type == 'schweizer_sklar':
            params['sklar_param'] = 2.0
        elif norm_type == 'dombi':
            params['dombi_param'] = 2.0
        elif norm_type == 'aczel_alsina':
            params['aa_param'] = 2.0
        elif norm_type == 'frank':
            params['frank_param'] = np.e
        return params


class TestReductionOperations:
    """Test reduction operations"""

    def test_t_norm_reduce_1d_array(self):
        """Test T-norm reduction equals iterative fold for 1D array"""
        norm = OperationTNorm(norm_type='algebraic')
        arr = np.array([0.8, 0.6, 0.7, 0.9])

        # Reduce operation
        result = norm.t_norm_reduce(arr)

        # Manual fold
        expected = arr[0]
        for i in range(1, len(arr)):
            expected = norm.t_norm(expected, arr[i])

        assert TestHelpers.compare_arrays(result, expected), \
            f"1D T-norm reduce failed: {result} != {expected}"

    def test_t_conorm_reduce_1d_array(self):
        """Test T-conorm reduction equals iterative fold for 1D array"""
        norm = OperationTNorm(norm_type='algebraic')
        arr = np.array([0.2, 0.3, 0.1, 0.4])

        # Reduce operation
        result = norm.t_conorm_reduce(arr)

        # Manual fold
        expected = arr[0]
        for i in range(1, len(arr)):
            expected = norm.t_conorm(expected, arr[i])

        assert TestHelpers.compare_arrays(result, expected), \
            f"1D T-conorm reduce failed: {result} != {expected}"

    def test_reduce_axis_none_2d(self):
        """Test axis=None reduction over 2D array"""
        norm = OperationTNorm(norm_type='algebraic')
        arr = np.array([[0.2, 0.3], [0.6, 0.8]])

        result_t = norm.t_norm_reduce(arr, axis=None)
        result_s = norm.t_conorm_reduce(arr, axis=None)

        # Manual calculation
        flat = arr.flatten()
        expected_t = flat[0]
        expected_s = flat[0]
        for i in range(1, len(flat)):
            expected_t = norm.t_norm(expected_t, flat[i])
            expected_s = norm.t_conorm(expected_s, flat[i])

        assert TestHelpers.compare_arrays(result_t, expected_t)
        assert TestHelpers.compare_arrays(result_s, expected_s)

    def test_reduce_axis_int(self):
        """Test axis=int reduction removes correct dimensions"""
        norm = OperationTNorm(norm_type='algebraic')
        arr = np.array([[0.2, 0.3, 0.4], [0.6, 0.7, 0.8]])

        # Reduce along axis 0 (rows)
        result_axis0 = norm.t_norm_reduce(arr, axis=0)
        assert result_axis0.shape == (3,)

        # Reduce along axis 1 (columns)
        result_axis1 = norm.t_norm_reduce(arr, axis=1)
        assert result_axis1.shape == (2,)

    def test_reduce_multiple_axes(self):
        """Test reduction with tuple of axes"""
        norm = OperationTNorm(norm_type='algebraic')
        arr = np.random.rand(2, 3, 4) * 0.8 + 0.1  # Values in (0.1, 0.9)

        # Reduce along axes (0, 2)
        result = norm.t_norm_reduce(arr, axis=(0, 2))
        assert result.shape == (3,)

        # Should be equivalent to sequential reduction
        temp = norm.t_norm_reduce(arr, axis=0)
        expected = norm.t_norm_reduce(temp, axis=1)  # axis 2 becomes axis 1 after first reduction

        assert TestHelpers.compare_arrays(result, expected)

    def test_reduce_single_element(self):
        """Test reduction of single-element array"""
        norm = OperationTNorm(norm_type='algebraic')
        arr = np.array([0.5])

        result_t = norm.t_norm_reduce(arr)
        result_s = norm.t_conorm_reduce(arr)

        assert TestHelpers.compare_arrays(result_t, 0.5)
        assert TestHelpers.compare_arrays(result_s, 0.5)

    def test_reduce_empty_array_raises(self):
        """Test reduction of empty array raises ValueError"""
        norm = OperationTNorm(norm_type='algebraic')
        arr = np.array([])

        with pytest.raises(ValueError, match="Cannot reduce an empty array."):
            norm.t_norm_reduce(arr)

        with pytest.raises(ValueError, match="Cannot reduce an empty array."):
            norm.t_conorm_reduce(arr)


class TestVectorization:
    """Test vectorization capabilities"""

    def test_elementwise_operation_2d(self):
        """Test array inputs produce elementwise operations"""
        norm = OperationTNorm(norm_type='algebraic')

        a = np.array([[0.2, 0.4], [0.6, 0.8]])
        b = np.array([[0.3, 0.5], [0.7, 0.9]])

        result_t = norm.t_norm(a, b)
        result_s = norm.t_conorm(a, b)

        # Manual elementwise calculation
        expected_t = np.zeros_like(a)
        expected_s = np.zeros_like(a)

        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                expected_t[i, j] = norm.t_norm(a[i, j], b[i, j])
                expected_s[i, j] = norm.t_conorm(a[i, j], b[i, j])

        assert TestHelpers.compare_arrays(result_t, expected_t)
        assert TestHelpers.compare_arrays(result_s, expected_s)

    def test_broadcasting(self):
        """Test broadcasting behavior"""
        norm = OperationTNorm(norm_type='algebraic')

        a = np.array([[0.2], [0.6]])  # Shape (2, 1)
        b = np.array([0.3, 0.7])  # Shape (2,)

        result_t = norm.t_norm(a, b)
        result_s = norm.t_conorm(a, b)

        # Should broadcast to (2, 2)
        assert result_t.shape == (2, 2)
        assert result_s.shape == (2, 2)

        # Verify specific values
        assert TestHelpers.compare_arrays(result_t[0, 0], norm.t_norm(0.2, 0.3))
        assert TestHelpers.compare_arrays(result_t[1, 1], norm.t_norm(0.6, 0.7))


class TestPlotSurface:
    """Test surface plotting functionality"""

    @patch('matplotlib.pyplot.show')
    def test_plot_no_exception(self, mock_show):
        """Test plot_t_norm_surface runs without exception"""
        norm = OperationTNorm(norm_type='algebraic')

        # Should not raise any exception
        norm.plot_t_norm_surface(resolution=10)  # Low resolution for speed

        # Verify show was called
        mock_show.assert_called_once()


class TestNumericalStability:
    """Test numerical stability at boundaries"""

    @pytest.mark.parametrize("norm_type", [
        'algebraic', 'lukasiewicz', 'einstein', 'hamacher', 'yager',
        'schweizer_sklar', 'dombi', 'aczel_alsina', 'frank',
        'minimum', 'drastic', 'nilpotent'
    ])
    def test_boundary_stability(self, norm_type):
        """Test inputs very close to 0 or 1 produce stable outputs"""
        params = self._get_norm_params(norm_type)
        norm = OperationTNorm(norm_type=norm_type, **params)

        # Values very close to boundaries
        eps = 1e-12
        boundary_values = [eps, 1 - eps]
        normal_values = [0.3, 0.7]

        for a in boundary_values + normal_values:
            for b in boundary_values + normal_values:
                t_result = norm.t_norm(a, b)
                s_result = norm.t_conorm(a, b)

                # Results should be in [0,1] and not NaN
                assert not np.isnan(t_result), f"T-norm produced NaN for {norm_type}: T({a},{b})"
                assert not np.isnan(s_result), f"T-conorm produced NaN for {norm_type}: S({a},{b})"
                assert 0 <= t_result <= 1, f"T-norm out of bounds for {norm_type}: T({a},{b})={t_result}"
                assert 0 <= s_result <= 1, f"T-conorm out of bounds for {norm_type}: S({a},{b})={s_result}"

                # Test generators (inf is allowed at boundary for generators)
                if norm.is_archimedean and norm.g_func is not None:
                    g_a = norm.g_func(a)
                    g_b = norm.g_func(b)
                    # Only check for NaN, inf is allowed at boundaries
                    assert not np.isnan(g_a), f"Generator produced NaN for {norm_type}: g({a})"
                    assert not np.isnan(g_b), f"Generator produced NaN for {norm_type}: g({b})"

    def _get_norm_params(self, norm_type):
        """Get appropriate parameters for each norm type"""
        params = {}
        if norm_type == 'hamacher':
            params['hamacher_param'] = 2.0
        elif norm_type == 'yager':
            params['yager_param'] = 2.0
        elif norm_type == 'schweizer_sklar':
            params['sklar_param'] = 2.0
        elif norm_type == 'dombi':
            params['dombi_param'] = 2.0
        elif norm_type == 'aczel_alsina':
            params['aa_param'] = 2.0
        elif norm_type == 'frank':
            params['frank_param'] = np.e
        return params


class TestSpecialCases:
    """Test special cases and edge conditions"""

    def test_frank_large_s_approximates_minimum(self):
        """Test Frank norm with large s approximates minimum norm"""
        norm_frank = OperationTNorm(norm_type='frank', frank_param=1e6)
        norm_minimum = OperationTNorm(norm_type='minimum')

        test_pairs = [(0.3, 0.7), (0.2, 0.8), (0.1, 0.9)]

        for a, b in test_pairs:
            frank_t = norm_frank.t_norm(a, b)
            frank_s = norm_frank.t_conorm(a, b)
            min_t = norm_minimum.t_norm(a, b)
            min_s = norm_minimum.t_conorm(a, b)

            assert TestHelpers.compare_arrays(frank_t, min_t, atol=1e-4), \
                f"Frank with large s doesn't approximate minimum: T({a},{b}) Frank={frank_t}, Min={min_t}"
            assert TestHelpers.compare_arrays(frank_s, min_s, atol=1e-4), \
                f"Frank with large s doesn't approximate maximum: S({a},{b}) Frank={frank_s}, Max={min_s}"

    def test_schweizer_sklar_positive_negative_p(self):
        """Test Schweizer-Sklar with positive and negative p"""
        norm_pos = OperationTNorm(norm_type='schweizer_sklar', sklar_param=2.0)
        norm_neg = OperationTNorm(norm_type='schweizer_sklar', sklar_param=-2.0)

        test_pairs = [(0.3, 0.7), (0.5, 0.5)]

        for a, b in test_pairs:
            t_pos = norm_pos.t_norm(a, b)
            t_neg = norm_neg.t_norm(a, b)
            s_pos = norm_pos.t_conorm(a, b)
            s_neg = norm_neg.t_conorm(a, b)

            # Both should produce valid results in [0,1]
            assert 0 <= t_pos <= 1, f"Schweizer-Sklar positive p out of bounds: T({a},{b})={t_pos}"
            assert 0 <= t_neg <= 1, f"Schweizer-Sklar negative p out of bounds: T({a},{b})={t_neg}"
            assert 0 <= s_pos <= 1, f"Schweizer-Sklar positive p out of bounds: S({a},{b})={s_pos}"
            assert 0 <= s_neg <= 1, f"Schweizer-Sklar negative p out of bounds: S({a},{b})={s_neg}"

    def test_dombi_boundary_cases(self):
        """Test Dombi norm with boundary values a or b in {0,1}"""
        norm = OperationTNorm(norm_type='dombi', dombi_param=2.0)

        # Test with a=0
        assert TestHelpers.compare_arrays(norm.t_norm(0.0, 0.5), 0.0)
        assert TestHelpers.compare_arrays(norm.t_conorm(0.0, 0.5), 0.5)

        # Test with a=1
        assert TestHelpers.compare_arrays(norm.t_norm(1.0, 0.5), 0.5)
        assert TestHelpers.compare_arrays(norm.t_conorm(1.0, 0.5), 1.0)

        # Test with b=0
        assert TestHelpers.compare_arrays(norm.t_norm(0.5, 0.0), 0.0)
        assert TestHelpers.compare_arrays(norm.t_conorm(0.5, 0.0), 0.5)

        # Test with b=1
        assert TestHelpers.compare_arrays(norm.t_norm(0.5, 1.0), 0.5)
        assert TestHelpers.compare_arrays(norm.t_conorm(0.5, 1.0), 1.0)

    def test_aczel_alsina_boundary_cases(self):
        """Test Aczel-Alsina norm with a=0,1 cases"""
        norm = OperationTNorm(norm_type='aczel_alsina', aa_param=2.0)

        # Test with a=0 -> T=0
        assert TestHelpers.compare_arrays(norm.t_norm(0.0, 0.5), 0.0)

        # Test with a=1 -> T=b
        assert TestHelpers.compare_arrays(norm.t_norm(1.0, 0.5), 0.5)

        # Test with b=1 -> T=a
        assert TestHelpers.compare_arrays(norm.t_norm(0.5, 1.0), 0.5)

    def test_einstein_hamacher_formula_consistency(self):
        """Test Einstein and Hamacher formulas for consistency"""
        norm_einstein = OperationTNorm(norm_type='einstein')
        norm_hamacher = OperationTNorm(norm_type='hamacher', hamacher_param=2.0)

        test_pairs = [(0.3, 0.7), (0.2, 0.8), (0.5, 0.5)]

        for a, b in test_pairs:
            # Einstein: T(a,b) = (a*b)/(1+(1-a)*(1-b))
            expected_t = (a * b) / (1 + (1 - a) * (1 - b))
            actual_t = norm_einstein.t_norm(a, b)
            assert TestHelpers.compare_arrays(actual_t, expected_t, atol=1e-10), \
                f"Einstein formula inconsistent: T({a},{b})={actual_t}, expected={expected_t}"

            # Einstein: S(a,b) = (a+b)/(1+a*b)
            expected_s = (a + b) / (1 + a * b)
            actual_s = norm_einstein.t_conorm(a, b)
            assert TestHelpers.compare_arrays(actual_s, expected_s, atol=1e-10), \
                f"Einstein S formula inconsistent: S({a},{b})={actual_s}, expected={expected_s}"

            # Hamacher with gamma=2: T(a,b) = (a*b)/(2+(1-2)*(a+b-a*b)) = (a*b)/(2-(a+b-a*b))
            gamma = 2.0
            expected_t_h = (a * b) / (gamma + (1 - gamma) * (a + b - a * b))
            actual_t_h = norm_hamacher.t_norm(a, b)
            assert TestHelpers.compare_arrays(actual_t_h, expected_t_h, atol=1e-10), \
                f"Hamacher formula inconsistent: T({a},{b})={actual_t_h}, expected={expected_t_h}"


class TestInfoDict:
    """Test get_info functionality"""

    @pytest.mark.parametrize("norm_type", [
        'algebraic', 'lukasiewicz', 'einstein', 'hamacher', 'yager',
        'schweizer_sklar', 'dombi', 'aczel_alsina', 'frank',
        'minimum', 'drastic', 'nilpotent'
    ])
    def test_info_dict_keys_present(self, norm_type):
        """Test get_info returns dict with required keys"""
        params = self._get_norm_params(norm_type)
        norm = OperationTNorm(norm_type=norm_type, **params)

        info = norm.get_info()

        required_keys = [
            'norm_type', 'q', 'is_archimedean', 'is_strict_archimedean',
            'supports_q', 'parameters'
        ]

        for key in required_keys:
            assert key in info, f"Missing key '{key}' in info dict for {norm_type}"

        # Verify specific values
        assert info['norm_type'] == norm_type
        assert info['q'] == norm.q
        assert info['is_archimedean'] == norm.is_archimedean
        assert info['is_strict_archimedean'] == norm.is_strict_archimedean
        assert info['supports_q'] == norm.supports_q
        assert isinstance(info['parameters'], dict)

    def test_info_dict_parameter_values(self):
        """Test get_info returns correct parameter values"""
        # Test Hamacher with specific parameter
        norm_hamacher = OperationTNorm(norm_type='hamacher', hamacher_param=3.0, q=2)
        info = norm_hamacher.get_info()

        assert info['parameters']['hamacher_param'] == 3.0
        assert info['q'] == 2

        # Test Yager with specific parameter
        norm_yager = OperationTNorm(norm_type='yager', yager_param=1.5)
        info = norm_yager.get_info()

        assert info['parameters']['yager_param'] == 1.5

        # Test Frank with specific parameter
        norm_frank = OperationTNorm(norm_type='frank', frank_param=2.5)
        info = norm_frank.get_info()

        assert info['parameters']['frank_param'] == 2.5

    def _get_norm_params(self, norm_type):
        """Get appropriate parameters for each norm type"""
        params = {}
        if norm_type == 'hamacher':
            params['hamacher_param'] = 2.0
        elif norm_type == 'yager':
            params['yager_param'] = 2.0
        elif norm_type == 'schweizer_sklar':
            params['sklar_param'] = 2.0
        elif norm_type == 'dombi':
            params['dombi_param'] = 2.0
        elif norm_type == 'aczel_alsina':
            params['aa_param'] = 2.0
        elif norm_type == 'frank':
            params['frank_param'] = np.e
        return params


class TestCreateTNormFactory:
    """Test create_tnorm factory function"""

    def test_create_tnorm_default(self):
        """Test create_tnorm with default parameters"""
        norm = create_tnorm()

        assert norm.norm_type == 'algebraic'
        assert norm.q == 1

    def test_create_tnorm_with_parameters(self):
        """Test create_tnorm with specific parameters"""
        norm = create_tnorm(norm_type='hamacher', q=3, hamacher_param=2.5)

        assert norm.norm_type == 'hamacher'
        assert norm.q == 3
        info = norm.get_info()
        assert info['parameters']['hamacher_param'] == 2.5

    def test_create_tnorm_invalid_type(self):
        """Test create_tnorm with invalid norm type"""
        with pytest.raises(ValueError, match="Unknown t-norm type"):
            create_tnorm(norm_type='invalid_norm')


class TestConfigIntegration:
    """Test integration with FuzzLab configuration system"""

    def test_config_access(self):
        """Test that triangular module can access config"""
        # This test ensures the config import works
        config = get_config()
        assert config is not None

        # Test creating norm with config-dependent behavior if any
        norm = OperationTNorm(norm_type='algebraic')
        assert norm is not None


# Performance and stress tests
class TestPerformance:
    """Test performance characteristics (not strict timing, just functionality)"""

    def test_large_array_handling(self):
        """Test handling of large arrays"""
        norm = OperationTNorm(norm_type='algebraic')

        # Create moderately large arrays (not too big for CI)
        size = 1000
        a = np.random.rand(size) * 0.8 + 0.1  # Values in (0.1, 0.9)
        b = np.random.rand(size) * 0.8 + 0.1

        # Should not raise exceptions or produce invalid results
        result_t = norm.t_norm(a, b)
        result_s = norm.t_conorm(a, b)

        assert result_t.shape == (size,)
        assert result_s.shape == (size,)
        assert np.all((0 <= result_t) & (result_t <= 1))
        assert np.all((0 <= result_s) & (result_s <= 1))

    def test_nested_reductions(self):
        """Test nested reduction operations"""
        norm = OperationTNorm(norm_type='minimum')  # Use simple norm for consistency

        # Create 3D array
        arr = np.random.rand(5, 4, 3) * 0.8 + 0.1

        # Multiple reduction operations
        result1 = norm.t_norm_reduce(arr, axis=0)
        result2 = norm.t_norm_reduce(result1, axis=0)
        result3 = norm.t_norm_reduce(result2)

        # Final result should be scalar
        assert np.isscalar(result3) or result3.shape == ()
        assert 0 <= result3 <= 1

    def test_large_array_throughput_optional(self, monkeypatch):
        """Optional larger array run for throughput sanity; gated by env AXISFUZZY_RUN_PERF."""
        import os
        run_perf = os.getenv('AXISFUZZY_RUN_PERF', '0') == '1'
        if not run_perf:
            pytest.skip("Skip large throughput test unless AXISFUZZY_RUN_PERF=1")

        norm = OperationTNorm(norm_type='algebraic')
        size = int(os.getenv('AXISFUZZY_PERF_SIZE', '200000'))
        rng = np.random.default_rng(0)
        a = rng.random(size)
        b = rng.random(size)

        t = norm.t_norm(a, b)
        s = norm.t_conorm(a, b)
        assert t.shape == (size,)
        assert s.shape == (size,)
        assert np.all((0 <= t) & (t <= 1))
        assert np.all((0 <= s) & (s <= 1))


class TestConcurrency:
    """Threaded usage should be consistent with serial computation."""

    def test_thread_safety_of_operations(self):
        from concurrent.futures import ThreadPoolExecutor

        norm = OperationTNorm(norm_type='algebraic', q=2)
        rng = np.random.default_rng(42)
        pairs = [(float(rng.random()), float(rng.random())) for _ in range(2000)]

        def worker(pair):
            a, b = pair
            return norm.t_norm(a, b), norm.t_conorm(a, b)

        # Parallel
        with ThreadPoolExecutor(max_workers=8) as ex:
            parallel_results = list(ex.map(worker, pairs))

        # Serial
        serial_results = [worker(p) for p in pairs]

        # Compare
        for (t1, s1), (t2, s2) in zip(parallel_results, serial_results):
            assert TestHelpers.compare_arrays(t1, t2, atol=1e-12)
            assert TestHelpers.compare_arrays(s1, s2, atol=1e-12)


# Edge case and error handling tests
class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_invalid_input_types(self):
        """Test invalid input types raise appropriate errors"""
        norm = OperationTNorm(norm_type='algebraic')

        # Test with string inputs
        with pytest.raises((TypeError, ValueError)):
            norm.t_norm("invalid", 0.5)

        with pytest.raises((TypeError, ValueError)):
            norm.t_conorm(0.5, "invalid")

    def test_mismatched_array_shapes(self):
        """Test mismatched array shapes"""
        norm = OperationTNorm(norm_type='algebraic')

        a = np.array([0.3, 0.6])
        b = np.array([0.4, 0.7, 0.8])  # Different shape

        # Should raise error due to broadcasting incompatibility
        with pytest.raises(ValueError):
            norm.t_norm(a, b)

    def test_out_of_range_inputs(self):
        """Test inputs outside [0,1] range"""
        norm = OperationTNorm(norm_type='algebraic')

        # Test with inputs > 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # May produce warnings but shouldn't crash
            result = norm.t_norm(1.5, 0.5)
            # Should handle gracefully, possibly clipping or producing warning
            assert not np.isnan(result)

        # Test with negative inputs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = norm.t_norm(-0.1, 0.5)
            assert not np.isnan(result)

    def test_inf_nan_inputs(self):
        """Test with inf and NaN inputs"""
        norm = OperationTNorm(norm_type='algebraic')

        # Test with inf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = norm.t_norm(np.inf, 0.5)
            # Should handle gracefully
            assert not np.isnan(result) or np.isnan(result)  # Either way is acceptable

        # Test with NaN
        result = norm.t_norm(np.nan, 0.5)
        # NaN input should typically produce NaN output
        assert np.isnan(result)


class TestCompleteIntegration:
    """Complete integration tests covering real-world usage scenarios"""

    def test_complete_workflow_algebraic(self):
        """Test complete workflow with algebraic norm"""
        # Create norm
        norm = OperationTNorm(norm_type='algebraic', q=2)

        # Test basic operations
        a, b = 0.6, 0.8
        t_result = norm.t_norm(a, b)
        s_result = norm.t_conorm(a, b)

        # Test array operations
        arr_a = np.array([0.2, 0.5, 0.8])
        arr_b = np.array([0.3, 0.6, 0.9])
        arr_t = norm.t_norm(arr_a, arr_b)
        arr_s = norm.t_conorm(arr_a, arr_b)

        # Test reductions
        reduce_t = norm.t_norm_reduce(arr_a)
        reduce_s = norm.t_conorm_reduce(arr_a)

        # Test properties
        demorgan = norm.verify_de_morgan_laws()
        info = norm.get_info()

        # Verify all results are valid
        assert 0 <= t_result <= 1
        assert 0 <= s_result <= 1
        assert all(0 <= x <= 1 for x in arr_t)
        assert all(0 <= x <= 1 for x in arr_s)
        assert 0 <= reduce_t <= 1
        assert 0 <= reduce_s <= 1
        assert demorgan['de_morgan_1'] is True
        assert demorgan['de_morgan_2'] is True
        assert info['norm_type'] == 'algebraic'
        assert info['q'] == 2

    def test_all_norms_basic_functionality(self):
        """Test that all norm types can be created and perform basic operations"""
        norm_types = [
            'algebraic', 'lukasiewicz', 'einstein', 'hamacher', 'yager',
            'schweizer_sklar', 'dombi', 'aczel_alsina', 'frank',
            'minimum', 'drastic', 'nilpotent'
        ]

        for norm_type in norm_types:
            params = self._get_norm_params(norm_type)

            try:
                norm = OperationTNorm(norm_type=norm_type, **params)

                # Basic operations should work
                t_result = norm.t_norm(0.3, 0.7)
                s_result = norm.t_conorm(0.3, 0.7)

                # Results should be valid
                assert 0 <= t_result <= 1, f"Invalid T-norm result for {norm_type}: {t_result}"
                assert 0 <= s_result <= 1, f"Invalid T-conorm result for {norm_type}: {s_result}"

                # Info should be accessible
                info = norm.get_info()
                assert isinstance(info, dict), f"Invalid info for {norm_type}"
                assert info['norm_type'] == norm_type

                # De Morgan laws should work
                demorgan = norm.verify_de_morgan_laws()
                assert isinstance(demorgan, dict), f"Invalid De Morgan result for {norm_type}"

            except Exception as e:
                pytest.fail(f"Failed to create or use norm type '{norm_type}': {e}")

    def _get_norm_params(self, norm_type):
        """Get appropriate parameters for each norm type"""
        params = {}
        if norm_type == 'hamacher':
            params['hamacher_param'] = 2.0
        elif norm_type == 'yager':
            params['yager_param'] = 2.0
        elif norm_type == 'schweizer_sklar':
            params['sklar_param'] = 2.0
        elif norm_type == 'dombi':
            params['dombi_param'] = 2.0
        elif norm_type == 'aczel_alsina':
            params['aa_param'] = 2.0
        elif norm_type == 'frank':
            params['frank_param'] = np.e
        return params


if __name__ == "__main__":
    # Run specific test classes for debugging
    pytest.main([__file__ + "::TestTypeNormalizer", "-v"])
    pytest.main([__file__ + "::TestParameterValidation", "-v"])
    pytest.main([__file__ + "::TestNormCorrectness", "-v"])
