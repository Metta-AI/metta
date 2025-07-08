# Map generation

## S3 maps

To produce maps in bulk and store them in S3, use the following commands:

### Creating maps

```bash
./tools/map/gen.py --output-uri=s3://BUCKET/DIR ./configs/env/mettagrid/map_builder/mapgen_auto.yaml
```

`mapgen_auto` builder is an example. You can use any YAML config that can be parsed by OmegaConf.

If `--output-uri` looks like a file (ends with `.yaml` or other extension), the map will be saved to that file.

Otherwise, the map will be saved to a file with a random suffix in that directory.

If `--output-uri` is not specified, the map won't be saved, only shown on screen.

To create maps in bulk, use `--count=N` option.

See `./tools/map/gen.py --help` for more options.

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

### Loading maps in map_builder configs

You can load a random map from an S3 directory in your YAML configs by using `metta.map.load_random.LoadRandom` as a map
builder.

`LoadRandom` allows you to modify the map by applying additional scenes to it. Check out
`configs/env/mettagrid/map_builder/load_random.yaml` for an example config that modifies the number of agents in the
map.
