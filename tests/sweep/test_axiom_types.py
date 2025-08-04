"""Unit tests for tAXIOM type utilities - MVP version."""

import pytest
from typing import Any

from metta.sweep.axiom.types import infer_type


class TestInferType:
    """Test the infer_type function."""

    def test_infer_type_with_annotations(self):
        """Test type inference from function annotations."""
        def typed_func(x: int) -> str:
            return str(x)
        
        input_type, output_type = infer_type(typed_func)
        assert input_type == int
        assert output_type == str

    def test_infer_type_partial_annotations(self):
        """Test type inference with partial annotations."""
        def input_only(x: float):
            return x * 2
        
        input_type, output_type = infer_type(input_only)
        assert input_type == float
        assert output_type is None
        
        def output_only(x) -> bool:
            return bool(x)
        
        input_type, output_type = infer_type(output_only)
        assert input_type is None
        assert output_type == bool

    def test_infer_type_no_annotations(self):
        """Test type inference with no annotations."""
        def untyped_func(x):
            return x
        
        input_type, output_type = infer_type(untyped_func)
        assert input_type is None
        assert output_type is None

    def test_infer_type_multiple_params(self):
        """Test type inference with multiple parameters."""
        def multi_param(x: int, y: str, z: float) -> bool:
            return True
        
        input_type, output_type = infer_type(multi_param)
        # Should infer from first parameter only
        assert input_type == int
        assert output_type == bool

    def test_infer_type_no_params(self):
        """Test type inference with no parameters."""
        def no_params() -> str:
            return "hello"
        
        input_type, output_type = infer_type(no_params)
        assert input_type is None
        assert output_type == str

    def test_infer_type_complex_types(self):
        """Test type inference with complex type annotations."""
        def complex_func(data: dict[str, Any]) -> list[int]:
            return [1, 2, 3]
        
        input_type, output_type = infer_type(complex_func)
        assert input_type == dict[str, Any]
        assert output_type == list[int]

    def test_infer_type_not_callable(self):
        """Test that non-callable returns None, None."""
        result = infer_type("not a function")
        assert result == (None, None)
        
        result = infer_type(42)
        assert result == (None, None)
        
        result = infer_type(None)
        assert result == (None, None)

    def test_infer_type_lambda(self):
        """Test type inference on lambda functions."""
        # Lambdas typically don't have annotations
        untyped_lambda = lambda x: x * 2
        input_type, output_type = infer_type(untyped_lambda)
        assert input_type is None
        assert output_type is None

    def test_infer_type_method(self):
        """Test type inference on class methods."""
        class MyClass:
            def method(self, x: str) -> int:
                return len(x)
        
        obj = MyClass()
        input_type, output_type = infer_type(obj.method)
        # Should skip 'self' and get x: str
        assert input_type == str
        assert output_type == int

    def test_infer_type_skips_self(self):
        """Test that type inference skips self parameter."""
        class TestClass:
            def no_args(self) -> str:
                return "test"
            
            def with_args(self, value: int) -> float:
                return float(value)
        
        obj = TestClass()
        
        # Method with no args after self
        input_type, output_type = infer_type(obj.no_args)
        assert input_type is None
        assert output_type == str
        
        # Method with args after self
        input_type, output_type = infer_type(obj.with_args)
        assert input_type == int
        assert output_type == float