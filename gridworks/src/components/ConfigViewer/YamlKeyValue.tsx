"use client";
import clsx from "clsx";
import { FC, use } from "react";

import { ConfigNode, isArrayNode, isObjectNode, isScalarNode } from "./utils";
import { YamlAny } from "./YamlAny";
import { YamlContext } from "./YamlContext";
import { YamlKey } from "./YamlKey";
import { YamlScalar } from "./YamlScalar";

export const YamlKeyValue: FC<{
  node: ConfigNode<Record<string, unknown>>;
  yamlKey: string;
}> = ({ yamlKey, node }) => {
  const value = node.value[yamlKey];

  const valueNode: ConfigNode = {
    value,
    path: [...node.path, yamlKey],
    depth: node.depth + 1,
  };

  const fullKey = valueNode.path.join(".");

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
        <YamlKey node={valueNode} />
        <YamlScalar node={valueNode} />
      </div>
    );
  }

  const singleLine =
    (isArrayNode(valueNode) && valueNode.value.length === 0) ||
    (isObjectNode(valueNode) && Object.keys(valueNode.value).length === 0);

  return (
    <div className={clsx(singleLine && "flex gap-1")}>
      <YamlKey node={valueNode} />
      <div className={clsx(!singleLine && "ml-[2ch]")}>
        <YamlAny node={valueNode} />
      </div>
    </div>
  );
};
