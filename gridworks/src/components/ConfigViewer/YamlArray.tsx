"use client";
import { FC } from "react";

import { ConfigNode } from "./utils";
import { YamlAny } from "./YamlAny";

export const YamlArray: FC<{
  node: ConfigNode<unknown[]>;
}> = ({ node }) => {
  if (node.value.length === 0) {
    return <div className="text-gray-500">[]</div>;
  }

  if (
    node.path[-1] === "map_data" &&
    node.value.every((v) => typeof v === "string" && v.length === 1)
  ) {
    // special case for readability - string[]
    return <div>{node.value.join("")}</div>;
  }

  return (
    <div className="">
      {node.value.map((v, i) => (
        <div key={`${node.path}.${i}`} className="flex gap-[1ch]">
          <div className="font-semibold text-blue-900">-</div>
          <YamlAny
            node={{
              value: v,
              path: [...node.path, String(i)],
              depth: node.depth + 1,
            }}
          />
        </div>
      ))}
    </div>
  );
};
