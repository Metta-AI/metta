"""Shared mutation conversion utilities for Python-to-C++ config conversion."""

from mettagrid.mettagrid_c import AlignmentMutationConfig as CppAlignmentMutationConfig
from mettagrid.mettagrid_c import AlignTo as CppAlignTo
from mettagrid.mettagrid_c import ClearInventoryMutationConfig as CppClearInventoryMutationConfig
from mettagrid.mettagrid_c import EntityRef as CppEntityRef
from mettagrid.mettagrid_c import FreezeMutationConfig as CppFreezeMutationConfig
from mettagrid.mettagrid_c import ResourceDeltaMutationConfig as CppResourceDeltaMutationConfig
from mettagrid.mettagrid_c import ResourceTransferMutationConfig as CppResourceTransferMutationConfig


def convert_entity_ref(target: str) -> CppEntityRef:
    """Convert Python entity target string to C++ EntityRef enum."""
    mapping = {
        "actor": CppEntityRef.actor,
        "target": CppEntityRef.target,
        "actor_collective": CppEntityRef.actor_collective,
        "target_collective": CppEntityRef.target_collective,
    }
    return mapping.get(target, CppEntityRef.target)


def convert_mutations(
    mutations: list,
    target_obj,
    resource_name_to_id: dict[str, int],
    limit_name_to_resource_ids: dict[str, list[int]],
    context: str = "",
) -> None:
    """Convert Python mutations and add them to a C++ config object.

    Args:
        mutations: List of Python mutation configs (AnyMutation)
        target_obj: C++ config object with add_*_mutation methods (e.g., CppHandlerConfig)
        resource_name_to_id: Dict mapping resource names to IDs
        limit_name_to_resource_ids: Dict mapping limit names to lists of resource IDs
        context: Description for error messages (e.g., "handler 'foo'")
    """
    for mutation in mutations:
        mutation_type = getattr(mutation, "mutation_type", None)

        if mutation_type == "resource_delta":
            # Resource delta mutation can have multiple deltas - add one mutation per resource
            for resource_name, delta in mutation.deltas.items():
                if resource_name in resource_name_to_id:
                    cpp_mutation = CppResourceDeltaMutationConfig()
                    cpp_mutation.entity = convert_entity_ref(mutation.target)
                    cpp_mutation.resource_id = resource_name_to_id[resource_name]
                    cpp_mutation.delta = delta
                    target_obj.add_resource_delta_mutation(cpp_mutation)

        elif mutation_type == "resource_transfer":
            # Resource transfer mutation can have multiple resources - add one mutation per resource
            for resource_name, amount in mutation.resources.items():
                if resource_name in resource_name_to_id:
                    cpp_mutation = CppResourceTransferMutationConfig()
                    cpp_mutation.source = convert_entity_ref(mutation.from_target)
                    cpp_mutation.destination = convert_entity_ref(mutation.to_target)
                    cpp_mutation.resource_id = resource_name_to_id[resource_name]
                    cpp_mutation.amount = amount
                    target_obj.add_resource_transfer_mutation(cpp_mutation)

        elif mutation_type == "alignment":
            cpp_mutation = CppAlignmentMutationConfig()
            if mutation.align_to == "actor_collective":
                cpp_mutation.align_to = CppAlignTo.actor_collective
            else:
                cpp_mutation.align_to = CppAlignTo.none
            target_obj.add_alignment_mutation(cpp_mutation)

        elif mutation_type == "freeze":
            cpp_mutation = CppFreezeMutationConfig()
            cpp_mutation.duration = mutation.duration
            target_obj.add_freeze_mutation(cpp_mutation)

        elif mutation_type == "clear_inventory":
            cpp_mutation = CppClearInventoryMutationConfig()
            cpp_mutation.entity = convert_entity_ref(mutation.target)
            limit_name = mutation.limit_name
            if limit_name not in limit_name_to_resource_ids:
                ctx_msg = f" in {context}" if context else ""
                raise ValueError(
                    f"ClearInventoryMutation{ctx_msg} references unknown limit_name '{limit_name}'. "
                    f"Available limits: {list(limit_name_to_resource_ids.keys())}"
                )
            cpp_mutation.resource_ids = limit_name_to_resource_ids[limit_name]
            target_obj.add_clear_inventory_mutation(cpp_mutation)
