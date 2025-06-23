"""
                          North\0
                      +---+---+---+
                      | 7 | 8 | 9 |
                      +---+---+---+
                      | 4 | 5 | 6 |
                      +---+---+---+
                      | 1 | 2 | 3 |
                      +---+---+---+
                            ●
       +---+---+---+                 +---+---+---+
       | 9 | 6 | 3 |                 | 1 | 4 | 7 |
       +---+---+---+                 +---+---+---+
West\2 | 8 | 5 | 2 | ●             ● | 2 | 5 | 8 | East/3
       +---+---+---+                 +---+---+---+
       | 7 | 4 | 1 |                 | 3 | 6 | 9 |
       +---+---+---+                 +---+---+---+
                            ●
                      +---+---+---+
                      | 3 | 2 | 1 |
                      +---+---+---+
                      | 6 | 5 | 4 |
                      +---+---+---+
                      | 9 | 8 | 7 |
                      +---+---+---+
                          South/1
"""


def attack_grid(orientation, idx):
    i = idx - 1
    if orientation == 0:
        dx = i % 3 - 1
        dy = -(i // 3) - 1
    elif orientation == 1:
        dx = -(i % 3) + 1
        dy = i // 3 + 1
    elif orientation == 2:
        dx = -(i // 3) - 1
        dy = -(i % 3) + 1
    elif orientation == 3:
        dx = i // 3 + 1
        dy = i % 3 - 1
    return (dx, dy)


for orientation in range(4):
    print(f"Orientation {orientation}:")
    for idx in range(1, 10):
        pos = attack_grid(orientation, idx)
        print(f"  {idx} -> {pos}")

assert attack_grid(0, 1) == (-1, -1)
assert attack_grid(0, 2) == (0, -1)
assert attack_grid(0, 3) == (1, -1)
assert attack_grid(0, 4) == (-1, -2)
assert attack_grid(0, 5) == (0, -2)
assert attack_grid(0, 6) == (1, -2)
assert attack_grid(0, 7) == (-1, -3)
assert attack_grid(0, 8) == (0, -3)
assert attack_grid(0, 9) == (1, -3)

assert attack_grid(1, 1) == (1, 1)
assert attack_grid(1, 2) == (0, 1)
assert attack_grid(1, 3) == (-1, 1)
assert attack_grid(1, 4) == (1, 2)
assert attack_grid(1, 5) == (0, 2)
assert attack_grid(1, 6) == (-1, 2)
assert attack_grid(1, 7) == (1, 3)
assert attack_grid(1, 8) == (0, 3)
assert attack_grid(1, 9) == (-1, 3)

assert attack_grid(2, 1) == (-1, 1)
assert attack_grid(2, 2) == (-1, 0)
assert attack_grid(2, 3) == (-1, -1)
assert attack_grid(2, 4) == (-2, 1)
assert attack_grid(2, 5) == (-2, 0)
assert attack_grid(2, 6) == (-2, -1)
assert attack_grid(2, 7) == (-3, 1)
assert attack_grid(2, 8) == (-3, 0)

assert attack_grid(3, 1) == (1, -1)
assert attack_grid(3, 2) == (1, 0)
assert attack_grid(3, 3) == (1, 1)
assert attack_grid(3, 4) == (2, -1)
assert attack_grid(3, 5) == (2, 0)
assert attack_grid(3, 6) == (2, 1)
assert attack_grid(3, 7) == (3, -1)
assert attack_grid(3, 8) == (3, 0)
assert attack_grid(3, 9) == (3, 1)
