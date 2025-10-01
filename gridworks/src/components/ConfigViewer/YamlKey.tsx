"use client";
import clsx from "clsx";
import { FC, use } from "react";

import { Tooltip } from "../Tooltip";
import { getSchema, getSchemaTypeStr, JsonSchema } from "./schema";
import { ConfigNode } from "./utils";
import { YamlContext } from "./YamlContext";

const JsonSchemaInfo: FC<{
  schema: JsonSchema;
}> = ({ schema }) => {
  let typeStr = getSchemaTypeStr(schema);
  return (
    <div>
      <span className="font-semibold">Type:</span> {typeStr}
      <div>
        {"description" in schema && (
          <div className="mt-1 text-gray-500">{schema.description}</div>
        )}
      </div>
    </div>
  );
};

const YamlKeyTooltip: FC<{ node: ConfigNode }> = ({ node }) => {
  const { kind } = use(YamlContext);
  const schema: JsonSchema | undefined = kind
    ? getSchema(node, kind)
    : undefined;

  return (
    <div className="max-w-[80ch] text-xs">
      <div>{node.path.join(".")}</div>
      {schema ? (
        <div className="mt-2 border-t border-gray-200 pt-2">
          <JsonSchemaInfo schema={schema} />
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
    <Tooltip render={() => <YamlKeyTooltip node={node} />}>{result}</Tooltip>
  );
};
