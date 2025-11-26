## Integration Checklist

### Imports
- [ ] Tool base class location
- [ ] Config base classes
- [ ] Stats functions (accumulate_rollout_stats)
- [ ] Curriculum classes

### Configuration
- [ ] SimulatorConfig fields match expected types
- [ ] CurriculumLPConfig fields match LearningProgressConfig
- [ ] SimulationConfig has all required fields

### Stats/Logging
- [ ] accumulate_rollout_stats() signature matches
- [ ] Metric prefixes match (env_, overview/, metric/)
- [ ] WandB logging uses correct API
- [ ] Gini coefficients logged at each layer

### Runtime
- [ ] Recipe runs without errors
- [ ] Metrics appear in WandB
- [ ] Output structure matches Sweep 2
- [ ] Performance curves look reasonable
