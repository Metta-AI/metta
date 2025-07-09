"""Tests for the instantiate function."""


# TODO: Update this import once instantiate.py is added
# from metta.common.util.instantiate import instantiate


class TestInstantiate:
    """Test cases for the instantiate function.

    This is a placeholder test file for the instantiate function that will be added
    in a separate PR. Once the function is available, uncomment the import and tests.
    """

    def test_instantiate_basic(self):
        """Test basic instantiation without recursion."""
        # config = DictConfig({
        #     "_target_": "builtins.dict",
        #     "a": 1,
        #     "b": 2
        # })
        # result = instantiate(config)
        # assert result == {"a": 1, "b": 2}
        pass

    def test_instantiate_with_kwargs(self):
        """Test instantiation with additional kwargs."""
        # config = DictConfig({
        #     "_target_": "builtins.dict",
        #     "a": 1
        # })
        # result = instantiate(config, b=2, c=3)
        # assert result == {"a": 1, "b": 2, "c": 3}
        pass

    def test_instantiate_recursive(self):
        """Test recursive instantiation."""
        # config = DictConfig({
        #     "_target_": "builtins.dict",
        #     "nested": {
        #         "_target_": "builtins.list",
        #         "_args_": [[1, 2, 3]]
        #     }
        # })
        # result = instantiate(config, _recursive_=True)
        # assert isinstance(result["nested"], list)
        # assert result["nested"] == [1, 2, 3]
        pass

    def test_instantiate_non_recursive(self):
        """Test that non-recursive mode doesn't instantiate nested configs."""
        # config = DictConfig({
        #     "_target_": "builtins.dict",
        #     "nested": {
        #         "_target_": "builtins.list",
        #         "_args_": [[1, 2, 3]]
        #     }
        # })
        # result = instantiate(config, _recursive_=False)
        # assert isinstance(result["nested"], DictConfig)
        # assert "_target_" in result["nested"]
        pass

    def test_instantiate_with_dict_input(self):
        """Test instantiation with plain dict input."""
        # config = {
        #     "_target_": "builtins.dict",
        #     "a": 1,
        #     "b": 2
        # }
        # result = instantiate(config)
        # assert result == {"a": 1, "b": 2}
        pass

    def test_instantiate_class_with_args(self):
        """Test instantiation of a class with positional arguments."""
        # config = DictConfig({
        #     "_target_": "builtins.range",
        #     "_args_": [0, 10, 2]
        # })
        # result = instantiate(config)
        # assert list(result) == [0, 2, 4, 6, 8]
        pass

    def test_instantiate_missing_target(self):
        """Test that instantiation fails without _target_."""
        # config = DictConfig({
        #     "a": 1,
        #     "b": 2
        # })
        # with pytest.raises(ValueError):
        #     instantiate(config)
        pass
