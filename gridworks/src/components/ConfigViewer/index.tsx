"use client";
import { FC, useState } from "react";

import { Menu } from "./Menu";
import { YamlAny } from "./YamlAny";
import { YamlContext } from "./YamlContext";
import { configToYaml } from "./utils";

export const ConfigViewer: FC<{
  value: unknown;
  // The fields that weren't set explicitly (so they can still have their default values, not necessarily null).
  // The prop name mirrors `model_fields_set` in Pydantic.
  unsetFields?: string[];
  kind?: string;
  isSelected?: (key: string, value: string) => boolean;
  onSelectLine?: (key: string, value: string) => void;
}> = ({ value, isSelected, onSelectLine, unsetFields, kind }) => {
  const [showDebugInfo, setShowDebugInfo] = useState(false);
  const [showDefaultValues, setShowDefaultValues] = useState(true);

  function handleCopyAsYaml() {
    if (typeof navigator !== "undefined" && navigator.clipboard) {
      navigator.clipboard.writeText(
        configToYaml({
          value,
          path: [],
          depth: 0,
        })
      );
    }
  }

  return (
    <YamlContext
      value={{
        isSelected,
        onSelectLine,
        unsetFields: new Set(unsetFields ?? []),
        kind,
        topValue: value,
        showDefaultValues,
        setShowDefaultValues,
        showDebugInfo,
        setShowDebugInfo,
      }}
    >
      <div className="relative overflow-auto rounded border border-gray-200 bg-gray-50 p-4 font-mono text-xs">
        <YamlAny node={{ value, path: [], depth: 0 }} />
        <div className="absolute top-2 right-2">
          <Menu onCopyAsYaml={handleCopyAsYaml} />
        </div>
      </div>
    </YamlContext>
  );
};
