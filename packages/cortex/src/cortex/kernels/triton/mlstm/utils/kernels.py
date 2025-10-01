#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.


def is_power_of_2(n):
    assert isinstance(n, int)
    return (n & (n - 1)) == 0
