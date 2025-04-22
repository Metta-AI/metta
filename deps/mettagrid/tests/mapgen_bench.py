import time

import hydra

from mettagrid.map.mapgen import MapGen


def bench_scene_gen(scene: str):
    width = 100
    height = 100

    start = time.time()
    MapGen(width, height, root=scene).build()
    time_taken = time.time() - start
    print(f"{scene} {width}x{height}: {time_taken:.3f}s")


@hydra.main(version_base=None, config_path="../configs", config_name="test_basic")  # unused
def main(cfg):
    bench_scene_gen("/scenes/wfc/blob")
    bench_scene_gen("/scenes/wfc/simple")
    bench_scene_gen("/scenes/convchain/blob")
    bench_scene_gen("/scenes/convchain/c_shape")
    bench_scene_gen("/scenes/convchain/diagonal")


if __name__ == "__main__":
    main()
