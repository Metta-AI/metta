"use client";
import clsx from "clsx";
import { FC, use } from "react";

import { Tooltip } from "../Tooltip";
import { getSchemaTypeStr, JsonSchema, useNodeSchema } from "./schema";
import { ConfigNode } from "./utils";
import { YamlContext } from "./YamlContext";

const JsonSchemaInfo: FC<{
  schema: JsonSchema;
}> = ({ schema }) => {
  const typeStr = getSchemaTypeStr(schema);
  return (
    <div>
      <span className="font-semibold">Type:</span> {typeStr}
      {"deprecated" in schema && schema.deprecated && (
        <span className="text-red-700"> (deprecated)</span>
      )}
      <div>
        {"description" in schema && (
          <div className="mt-1 text-gray-500">{schema.description}</div>
        )}
      </div>
    </div>
  );
};

const YamlKeyTooltip: FC<{ node: ConfigNode }> = ({ node }) => {
  const { schema } = useNodeSchema(node);
  const { showDebugInfo } = use(YamlContext);

  return (
    <div className="max-w-[80ch] text-xs">
      <div>{node.path.join(".")}</div>
      {schema ? (
        <div className="mt-2 border-t border-gray-200 pt-2">
          <JsonSchemaInfo schema={schema} />
          {showDebugInfo && <pre>{JSON.stringify(schema, null, 2)}</pre>}
        </div>
      ) : null}
    </div>
  );
};

export const YamlKey: FC<{
  node: ConfigNode;
}> = ({ node }) => {
  const { unsetFields } = use(YamlContext);
  const disabled = unsetFields.has(node.path.join("."));
  const name = node.path[node.path.length - 1];

  const result = (
    <span
      className={clsx(
        disabled ? "text-gray-500" : "font-semibold text-blue-900"
      )}
    >
      <span className="cursor-pointer hover:bg-blue-100">{name}</span>:
    </span>
  );
  return (
    <div className="flex">
      <Tooltip render={() => <YamlKeyTooltip node={node} />}>{result}</Tooltip>
    </div>
  );
};
