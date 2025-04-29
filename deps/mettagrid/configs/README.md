Configs in this directory are loaded with OmegaConf, without Hydra. So you can't use `defaults` lists in them or other Hydra-specific features.

They are used in tests and in some parts of map generation code and scripts.

Configs under `./scenes` are used by `mettagrid.map.scene` to load scenes by reference: when the scene is referenced with `/wfc/blob.yaml`, it's resolved to `configs/scenes/wfc/blob.yaml` in mettagrid package.

Configs under `./maps` are used by mapgen CLI tests. Most of map configs are in Metta's `configs/env/mettagrid/game/map_builder/` directory.
