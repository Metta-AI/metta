"use client";
import { FC } from "react";

import { getSchemaTypeStr, JsonSchema } from "./schema";

export const JsonSchemaInfo: FC<{
  schema: JsonSchema;
}> = ({ schema }) => {
  let typeStr = getSchemaTypeStr(schema);
  return (
    <div className="max-w-[80ch] text-xs">
      <span className="font-semibold">Type:</span> {typeStr}
      <div>{"description" in schema && schema.description}</div>
    </div>
  );
};
