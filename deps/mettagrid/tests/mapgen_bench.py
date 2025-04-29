import time

from mettagrid.map.mapgen import MapGen


def bench_scene_gen(scene: str):
    width = 100
    height = 100

    start = time.time()
    MapGen(width, height, root=scene).build()
    time_taken = time.time() - start
    print(f"{scene} {width}x{height}: {time_taken:.3f}s")


def main():
    bench_scene_gen("wfc/blob.yaml")
    bench_scene_gen("wfc/simple.yaml")
    bench_scene_gen("convchain/blob.yaml")
    bench_scene_gen("convchain/c_shape.yaml")
    bench_scene_gen("convchain/diagonal.yaml")


if __name__ == "__main__":
    main()
