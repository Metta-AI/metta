
import numpy as np

from carbs import LinearSpace
from carbs import LogSpace
from carbs import LogitSpace
from carbs import Param


def carbs_param(group, name, space, wandb_params, mmin=None, mmax=None,
        search_center=None, is_integer=False, rounding_factor=1, scale=1):
    wandb_param = wandb_params[group]['parameters'][name]
    if 'values' in wandb_param:
        values = wandb_param['values']
        mmin = min(values)
        mmax = max(values)

    if mmin is None:
        mmin = float(wandb_param['min'])
    if mmax is None:
        mmax = float(wandb_param['max'])

    if space == 'log':
        Space = LogSpace
        if search_center is None:
            search_center = 2**(np.log2(mmin) + np.log2(mmax)/2)
    elif space == 'linear':
        Space = LinearSpace
        if search_center is None:
            search_center = (mmin + mmax)/2
    elif space == 'logit':
        Space = LogitSpace
        assert mmin == 0
        assert mmax == 1
        assert search_center is not None
    else:
        raise ValueError(f'Invalid CARBS space: {space} (log/linear)')

    return Param(
        name=f'{group}/{name}',
        space=Space(
            min=mmin,
            max=mmax,
            is_integer=is_integer,
            rounding_factor=rounding_factor,
            scale=scale,
        ),
        search_center=search_center,
    )
