from collections import Counter, deque
from typing import Dict, List, Optional


class Action:
    def __init__(self, name: str, inputs: Dict[str, int], outputs: Dict[str, int], action_type: str):
        self.name = name
        self.inputs = Counter(inputs)
        self.outputs = Counter(outputs)
        self.type = action_type  # "generator", "factory", or "altar"

    def __repr__(self):
        return f"{self.type.capitalize()} '{self.name}': {dict(self.inputs)} -> {dict(self.outputs)}"


class ResourceGraphSolver:
    @staticmethod
    def from_node_descriptions(node_descriptions: Dict[int, Dict]) -> "ResourceGraphSolver":
        actions = []
        for node_id, desc in node_descriptions.items():
            inputs = desc["input_colors"]
            outputs = desc["output_colors"]
            node_type = desc["node_type"]
            name = f"Node{node_id}"

            # Special case: altars output a Heart
            if node_type == "altar":
                outputs = {"Heart": 1}

            actions.append(Action(name, inputs, outputs, node_type))

        return ResourceGraphSolver(actions)

    def __init__(self, actions: List[Action]):
        self.actions = actions
        self.output_index: Dict[str, List[Action]] = {}
        for action in actions:
            for output in action.outputs:
                self.output_index.setdefault(output, []).append(action)

    def solve(self, goal: Dict[str, int]) -> Optional[List[str]]:
        _State = tuple[Counter, List[str]]  # (inventory, plan)
        visited = set()
        queue = deque()

        # Start with empty inventory and empty plan
        queue.append((Counter(), []))

        def inventory_key(inv: Counter) -> tuple:
            """Hashable inventory snapshot for visited set"""
            return tuple(sorted((k, v) for k, v in inv.items() if v > 0))

        while queue:
            inventory, plan = queue.popleft()

            # Check if goal is satisfied
            if all(inventory.get(res, 0) >= qty for res, qty in goal.items()):
                return plan

            # Skip if this inventory was seen
            inv_key = inventory_key(inventory)
            if inv_key in visited:
                continue
            visited.add(inv_key)

            # Try every action
            for action in self.actions:
                if all(inventory.get(res, 0) >= qty for res, qty in action.inputs.items()):
                    # Apply action
                    new_inventory = inventory.copy()
                    new_inventory.subtract(action.inputs)
                    new_inventory.update(action.outputs)

                    input_str = " + ".join(f"{v} {k}" for k, v in action.inputs.items()) or "nothing"
                    output_str = " + ".join(f"{v} {k}" for k, v in action.outputs.items())
                    new_plan = plan + [f"Use {action.type} '{action.name}': ({input_str}) → ({output_str})"]

                    queue.append((new_inventory, new_plan))

        # No plan found
        return None


# Example usage
if __name__ == "__main__":
    actions = [
        Action("OG1", {}, {"Orange": 1}, "generator"),
        Action("OG2", {}, {"Red": 1}, "generator"),
        Action("Factory1", {"Orange": 2, "Red": 1}, {"Red": 1, "Blue": 1}, "factory"),
        Action("HeartAltar", {"Red": 1, "Blue": 1, "Orange": 1}, {"Heart": 1}, "altar"),
    ]

    solver = ResourceGraphSolver(actions)
    plan = solver.solve({"Heart": 1})

    if plan:
        for step in plan:
            print(step)
    else:
        print("❌ No plan found.")
