#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import triton


def get_head_dim_block_size(head_dim: int, min_block_size: int = 64) -> int:
    return min(min_block_size, triton.next_power_of_2(head_dim))
