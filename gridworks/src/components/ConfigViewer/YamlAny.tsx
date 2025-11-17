"use client";
import { FC } from "react";

import { ConfigNode, isArrayNode, isObjectNode, isScalarNode } from "./utils";
import { YamlArray } from "./YamlArray";
import { YamlObject } from "./YamlObject";
import { YamlScalar } from "./YamlScalar";

export const YamlAny: FC<{
  node: ConfigNode;
  disabled?: boolean;
}> = ({ node, disabled }) => {
  if (isArrayNode(node)) {
    return <YamlArray node={node} />;
  }
  if (isObjectNode(node)) {
    return <YamlObject node={node} />;
  }

  if (isScalarNode(node)) {
    return <YamlScalar node={node} disabled={disabled} />;
  }

  if (node.value === undefined) {
    return "undefined??";
  }

  throw new Error(`Unknown value type: ${typeof node.value}`);
};
