# Map generation

## S3 maps

To produce maps in bulk and store them in S3, use the following commands:

### Creating maps

```bash
./packages/mettagrid/python/src/mettagrid/mapgen/tools/gen.py --output-uri=s3://BUCKET/DIR ./configs/env/mettagrid/map_builder/mapgen_auto.yaml
```

`mapgen_auto` builder is an example. You can use any YAML config that can be parsed by OmegaConf.

If `--output-uri` looks like a file (ends with `.yaml` or other extension), the map will be saved to that file.

Otherwise, the map will be saved to a file with a random suffix in that directory.

If `--output-uri` is not specified, the map won't be saved, only shown on screen.

To create maps in bulk, use `--count=N` option.

See `./packages/mettagrid/python/src/mettagrid/mapgen/tools/gen.py --help` for more options.

### Viewing maps

You can view a single map by running:

```bash
./tools/map/view.py s3://BUCKET/PATH/TO/MAP.yaml
```

The following command will show a random map from an S3 directory:

```bash
./tools/map/view.py s3://BUCKET/DIR
```

Same heuristics about detecting if the URI is a file apply here.
