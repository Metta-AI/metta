# Protein Sweep Tests

This directory contains comprehensive unit and integration tests for the protein sweep functionality in Metta.

## Test Files

### test_protein_serialization.py
Unit tests for WandB object serialization and numpy type cleaning:
- Tests `_deep_clean()` method handling of numpy types
- Tests handling of WandB SummarySubDict objects
- Tests mixed numpy and WandB type serialization
- Tests fallback to string representation for non-serializable objects
- Verifies JSON serializability of all saved data

### test_metta_protein.py
Unit tests for the MettaProtein class:
- Tests initialization with sweep configuration
- Tests parameter transformation and numpy cleaning
- Tests configuration defaults handling
- Tests WandB config override behavior
- Tests integration between Protein and WandB

### test_protein_integration.py
Integration tests for the full protein sweep pipeline:
- Tests loading observations from previous runs
- Tests recording new observations updates both WandB and Protein
- Tests realistic serialization scenarios with numpy and WandB objects
- Tests sweep state file JSON compatibility
- Tests various run states (success, failure, running, defunct)

### test_protein_observation_loading.py
Unit tests specifically for the critical observation loading fix:
- Tests that nested WandB suggestions are properly flattened before passing to Protein
- Tests handling of deeply nested parameter structures
- Tests error handling for malformed suggestions
- Tests that pufferlib.unroll_nested_dict is used for flattening
- Verifies the fix for the main bug where Protein wasn't learning from observations

### test_protein_fill_parameter.py
Unit tests for the protein fill parameter fix:
- Tests that `protein.suggest()` is always called with `fill=None`
- Verifies no TypeError occurs even with cleaned/string data
- Ensures the fill parameter fix prevents "'str' object does not support item assignment" errors
- Tests that _suggestion_info is properly stored but not misused as fill parameter

## Key Test Coverage

1. **Observation Loading Fix**: The main bug where Protein received nested dicts but expected flattened format is thoroughly tested
2. **Serialization**: All numpy types and WandB objects are properly cleaned before JSON serialization
3. **State Management**: Various run states and transitions are tested
4. **Error Handling**: Graceful handling of malformed data and network errors
5. **Integration**: Full pipeline from loading historical runs to generating new suggestions

## Running Tests

Run all sweep tests:
```bash
python -m pytest tests/sweep/ -v
```

Run specific test file:
```bash
python -m pytest tests/sweep/test_protein_observation_loading.py -v
```

Run specific test:
```bash
python -m pytest tests/sweep/test_protein_observation_loading.py::TestProteinObservationLoading::test_observation_loading_flattens_nested_suggestions -v
```
