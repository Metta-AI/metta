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

export function configToYaml(rootNode: ConfigNode): string {
  let output: string = "";

  function convert(node: ConfigNode): void {
    const indent = "  ".repeat(node.depth);

    if (node.value === null) {
      output += `${indent}null\n`;
      return;
    }

    if (isScalarNode(node)) {
      output += `${indent}${node.value}\n`;
      return;
    }

    if (isArrayNode(node)) {
      for (const item of node.value) {
        const child: ConfigNode = {
          path: [...node.path],
          depth: node.depth + 1,
          value: item,
        };

        // Special fast-paths for common simple array items
        if (item === null) {
          output += `${indent}- null\n`;
          continue;
        }
        if (isScalarNode(child)) {
          output += `${indent}- ${String(item)}\n`;
          continue;
        }
        if (isArrayNode(child) && child.value.length === 0) {
          output += `${indent}- []\n`;
          continue;
        }
        if (isObjectNode(child) && Object.keys(child.value).length === 0) {
          output += `${indent}- {}\n`;
          continue;
        }

        // No newline after dash for nested structures
        output += `${indent}-`;
        convert(child);
      }
      return;
    }

    if (isObjectNode(node)) {
      const entries = Object.entries(node.value);

      if (entries.length === 0) {
        return;
      }

      for (let i = 0; i < entries.length; i++) {
        const [key, value] = entries[i];
        // Do not indent the first key of an object in an array
        const prefix = i === 0 && output.endsWith("-") ? " " : `${indent}`;

        const child: ConfigNode = {
          path: [...node.path, key],
          depth: node.depth + 1,
          value,
        };

        // Fast-paths for common simple values
        if (isScalarNode(child)) {
          output += `${prefix}${key}: ${child.value}\n`;
          continue;
        }
        if (isArrayNode(child) && child.value.length === 0) {
          output += `${prefix}${key}: []\n`;
          continue;
        }
        if (isObjectNode(child) && Object.keys(child.value).length === 0) {
          output += `${prefix}${key}: {}\n`;
          continue;
        }

        output += `${prefix}${key}:\n`;
        convert(child);
      }
    }
  }

  convert(rootNode);
  return output;
}
