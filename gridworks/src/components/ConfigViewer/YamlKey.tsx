"use client";
import clsx from "clsx";
import { FC } from "react";

import { Tooltip } from "../Tooltip";
import { JsonSchemaInfo } from "./JsonSchemaInfo";
import { JsonSchema } from "./schema";

export const YamlKey: FC<{
  name: string;
  schema?: JsonSchema;
  disabled?: boolean;
}> = ({ name, disabled, schema }) => {
  let result = (
    <span
      className={clsx(
        disabled ? "text-gray-500" : "font-semibold text-blue-900"
      )}
    >
      <span className={clsx(schema && "cursor-pointer hover:bg-blue-100")}>
        {name}
      </span>
      :
    </span>
  );
  if (schema) {
    result = (
      <Tooltip render={() => <JsonSchemaInfo schema={schema} />}>
        {result}
      </Tooltip>
    );
  }
  return result;
};
