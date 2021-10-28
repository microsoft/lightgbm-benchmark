# Tutorial: Create a script and pipeline to run in AzureML

**Motivations** - The scripts of the lightgbm-benchmark rely on a helper class that is structuring and standardizing the behavior of each script (init/close of MLFlow, logging system metrics and properties, etc).

## A. Write a specification and test

### Edit your component specification in yaml

```yaml
```

### Adding your script to unit tests

```python
COMPONENT_SPEC_FILES = [
    "generate_data/generate_spec.yaml",
    "lightgbm_data2bin/data2bin_spec.yaml",
    "lightgbm_python/train_spec.yaml",

    # ... many others here

    # add your own as relative path from src/scripts/
    "my_component/spec.yaml"
]
```
