"use client";
import { FC } from "react";

import { YamlAny } from "./YamlAny";
import { YamlContext } from "./YamlContext";

export const ConfigViewer: FC<{
  value: unknown;
  // The fields that weren't set explicitly (so they can still have their default values, not necessarily null).
  // The prop name mirrors `model_fields_set` in Pydantic.
  unsetFields?: string[];
  kind?: string;
  isSelected?: (key: string, value: string) => boolean;
  onSelectLine?: (key: string, value: string) => void;
}> = ({ value, isSelected, onSelectLine, unsetFields, kind }) => {
  return (
    <YamlContext.Provider
      value={{
        isSelected,
        onSelectLine,
        unsetFields: new Set(unsetFields ?? []),
        kind,
        topValue: value,
      }}
    >
      <div className="overflow-auto rounded border border-gray-200 bg-gray-50 p-4 font-mono text-xs">
        <YamlAny node={{ value, path: [], depth: 0 }} />
      </div>
    </YamlContext.Provider>
  );
};
