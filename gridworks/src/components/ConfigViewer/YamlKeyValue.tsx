"use client";
import clsx from "clsx";
import { FC, use, useMemo, useState } from "react";

import { ConfigNode, isArrayNode, isObjectNode, isScalarNode } from "./utils";
import { YamlAny } from "./YamlAny";
import { YamlContext } from "./YamlContext";
import { YamlKey } from "./YamlKey";
import { YamlScalar } from "./YamlScalar";

export const YamlKeyValue: FC<{
  node: ConfigNode<Record<string, unknown>>;
  yamlKey: string;
}> = ({ yamlKey, node }) => {
  const { unsetFields, showDefaultValues } = use(YamlContext);
  const value = node.value[yamlKey];

  const valueNode: ConfigNode = {
    value,
    path: [...node.path, yamlKey],
    depth: node.depth + 1,
  };

  const fullKey = valueNode.path.join(".");

  const disabled = useMemo(() => {
    let path = "";
    for (const part of valueNode.path) {
      path = path ? `${path}.${part}` : part;
      if (unsetFields.has(path)) {
        return true;
      }
    }
    return false;
  }, [unsetFields, valueNode.path]);

  const [isExpanded, setIsExpanded] = useState(true);

  if (disabled && !showDefaultValues) {
    return null;
  }

  if (isScalarNode(valueNode)) {
    const { isSelected, onSelectLine } = use(YamlContext);

    const isActive = isSelected?.(fullKey, String(value));

    const onClick = onSelectLine
      ? () => onSelectLine(fullKey, String(value))
      : undefined;

    return (
      <div
        className={clsx(
          "flex gap-1",
          isActive && "-mx-1 bg-blue-100 px-1",
          onClick && "cursor-pointer hover:bg-blue-200"
        )}
        onClick={onClick}
      >
        <YamlKey node={valueNode} canExpand={false} disabled={disabled} />
        <YamlScalar node={valueNode} disabled={disabled} />
      </div>
    );
  }

  const singleLine =
    (isArrayNode(valueNode) && valueNode.value.length === 0) ||
    (isObjectNode(valueNode) && Object.keys(valueNode.value).length === 0);

  return (
    <div className={clsx(singleLine && "flex gap-1")}>
      <YamlKey
        node={valueNode}
        canExpand={!singleLine}
        isExpanded={isExpanded}
        disabled={disabled}
        onExpansionClick={() => setIsExpanded(!isExpanded)}
      />

      <div className={clsx(!singleLine && "ml-[2ch]", !isExpanded && "hidden")}>
        <YamlAny node={valueNode} />
      </div>
    </div>
  );
};
