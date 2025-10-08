"use client";
import { FC } from "react";

import { ConfigNode } from "./utils";
import { YamlKeyValue } from "./YamlKeyValue";

export const YamlObject: FC<{
  node: ConfigNode<Record<string, unknown>>;
}> = ({ node }) => {
  if (Object.keys(node.value).length === 0) {
    return <div className="text-gray-500">{"{}"}</div>;
  }

  return (
    <div>
      {Object.keys(node.value).map((key) => (
        <YamlKeyValue key={key} node={node} yamlKey={key} />
      ))}
    </div>
  );
};
