# Map generation

## S3 maps

To produce maps in bulk and store them in S3, use the following commands:

### Creating maps

```bash
python -m tools.map.gen --output-uri=s3://BUCKET/DIR ./configs/game/map_builder/mapgen_auto.yaml
```

`mapgen_auto` builder is an example. You can use any YAML config that can be parsed by OmegaConf.

If `--output-uri` looks like a file (ends with `.yaml` or other extension), the map will be saved to that file.

Otherwise, the map will be saved to a file with a random suffix in that directory.

If `--output-uri` is not specified, the map won't be saved, only shown on screen.

To create maps in bulk, use `--count=N` option.

See `python -m tools.map.gen --help` for more options.

### Viewing maps

You can view a single map by running:

```bash
python -m tools.map.view s3://BUCKET/PATH/TO/MAP.yaml
```

The following command will show a random map from an S3 directory:

```bash
python -m tools.map.view s3://BUCKET/DIR
```

Same heuristics about detecting if the URI is a file apply here.

### Loading maps in map_builder configs

You can load a random map from an S3 directory in your YAML configs by using `mettagrid.map.load_random.LoadRandom` as a map builder.

`LoadRandom` allows you to modify the map by applying additional scenes to it. Check out `configs/game/map_builder/load_random.yaml` for an example config that modifies the number of agents in the map.

### Indexing maps

Optionally, you can index your maps to make loading them faster.

This is intended to speed up reading from S3. It shouldn't change any functionality, and you should skip playing with this unless you find map loading from S3 is slow.

Index is a plain text file that lists URIs of all the maps. You can assemble it manually, or use the following script:

```bash
python -m tools.index_s3_maps --dir=s3://BUCKET/DIR --target=s3://BUCKET/DIR/index.txt
```

`--target` is optional. If not provided, the index will be saved to `{--dir}/index.txt`.

You can then use `mettagrid.map.load_random_from_index.LoadRandomFromIndex` to load a random map from the index.
