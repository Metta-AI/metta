export function extendPath(path: string, nextPart: string): string {
  return path ? `${path}.${nextPart}` : nextPart;
}

export function isScalarNode(
  node: ConfigNode
): node is ConfigNode<string | number | boolean | null> {
  return (
    typeof node.value === "string" ||
    typeof node.value === "number" ||
    typeof node.value === "boolean" ||
    node.value === null
  );
}

export type ConfigNode<T = unknown> = {
  path: string[];
  depth: number;
  value: T;
};

export function isArrayNode(node: ConfigNode): node is ConfigNode<unknown[]> {
  return Array.isArray(node.value);
}

export function isObjectNode(
  node: ConfigNode
): node is ConfigNode<Record<string, unknown>> {
  return typeof node.value === "object" && node.value !== null;
}
