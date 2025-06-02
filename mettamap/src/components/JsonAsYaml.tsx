"use client";
import clsx from "clsx";
import { FC } from "react";

const YamlPart: FC<{
  yamlKey: string;
  value: unknown;
  path: string;
  depth: number;
  isSelected?: (key: string, value: string) => boolean;
  onSelectLine?: (key: string, value: string) => void;
}> = ({ yamlKey, value, path, depth, isSelected, onSelectLine }) => {
  const fullKey = path ? `${path}.${yamlKey}` : yamlKey;

  if (typeof value === "object" && value !== null) {
    return (
      <div key={fullKey} className="ml-2">
        <div className="font-mono font-bold">{yamlKey}:</div>
        {Object.entries(value).map(([k, v]) => (
          <YamlPart
            key={`${fullKey}.${k}`}
            yamlKey={k}
            value={v}
            path={fullKey}
            depth={depth + 1}
            isSelected={isSelected}
            onSelectLine={onSelectLine}
          />
        ))}
      </div>
    );
  }

  const isActive = isSelected?.(fullKey, String(value));

  const multiline = typeof value === "string" && value.includes("\n");

  return (
    <div
      key={fullKey}
      className={clsx(
        // ml-1 + px-1 add up to same offset as ml-2 for object lines
        "ml-1 cursor-pointer rounded px-1 py-0.5 font-mono hover:bg-blue-100",
        isActive && "bg-blue-200"
      )}
      onClick={() => onSelectLine?.(fullKey, String(value))}
    >
      <span className="font-bold">{yamlKey}:</span> {multiline && "|"}
      <span className="whitespace-pre-wrap">
        {typeof value === "string" && value.includes("\n")
          ? "\n" + value
          : String(value)}
      </span>
    </div>
  );
};

export const JsonAsYaml: FC<{
  json: Record<string, unknown>;
  isSelected?: (key: string, value: string) => boolean;
  onSelectLine?: (key: string, value: string) => void;
}> = ({ json, isSelected, onSelectLine }) => {
  return (
    <div className="overflow-auto rounded border border-gray-200 bg-gray-50 p-4 text-xs">
      {Object.entries(json).map(([key, value]) => (
        <YamlPart
          key={key}
          yamlKey={key}
          value={value}
          path=""
          depth={0}
          isSelected={isSelected}
          onSelectLine={onSelectLine}
        />
      ))}
    </div>
  );
};
