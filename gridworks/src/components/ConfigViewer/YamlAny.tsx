"use client";
import { FC } from "react";

import { ConfigNode, isArrayNode, isObjectNode, isScalarNode } from "./utils";
import { YamlArray } from "./YamlArray";
import { YamlObject } from "./YamlObject";
import { YamlScalar } from "./YamlScalar";

export const YamlAny: FC<{
  node: ConfigNode;
}> = ({ node }) => {
  if (isArrayNode(node)) {
    return <YamlArray node={node} />;
  }
  if (isObjectNode(node)) {
    return <YamlObject node={node} />;
  }

  if (isScalarNode(node)) {
    return <YamlScalar node={node} />;
  }

  if (node.value === undefined) {
    return "undefined??";
  }

  throw new Error(`Unknown value type: ${typeof node.value}`);
};
