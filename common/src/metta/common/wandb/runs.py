import itertools
import typing

import wandb
import wandb.apis.public.runs

import metta.common.util.constants


def find_training_runs(
    wandb_tags: list[str] | None = None,
    author: str | None = None,
    state: str | None = None,
    created_after: str | None = None,
    created_before: str | None = None,
    entity: str = metta.common.util.constants.METTA_WANDB_ENTITY,
    project: str = metta.common.util.constants.METTA_WANDB_PROJECT,
    order_by: str = "-created_at",
    run_names: list[str] | None = None,
    limit: int = 50,
) -> typing.Iterable[wandb.apis.public.runs.Run]:
    filters = {}
    if state:
        filters["state"] = state
    if author:
        filters["username"] = author
    if created_after:
        filters["created_at"] = {"$gte": created_after}

    if created_before:
        if "created_at" in filters:
            filters["created_at"]["$lte"] = created_before
        else:
            filters["created_at"] = {"$lte": created_before}
    if wandb_tags:
        filters["tags"] = {"$in": wandb_tags}
    if run_names:
        filters["name"] = {"$in": run_names}
    return itertools.islice(wandb.Api().runs(f"{entity}/{project}", filters=filters, order=order_by), limit)
