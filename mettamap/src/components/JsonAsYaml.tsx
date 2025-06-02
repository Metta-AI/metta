"use client";
import clsx from "clsx";
import { FC } from "react";

export const JsonAsYaml: FC<{
  json: Record<string, unknown>;
  isSelected?: (key: string, value: string) => boolean;
  onSelectLine?: (key: string, value: string) => void;
}> = ({ json, isSelected, onSelectLine }) => {
  // Function to render a key-value pair as a clickable line
  const renderYamlLine = (key: string, value: unknown, path: string = "") => {
    const fullKey = path ? `${path}.${key}` : key;

    if (typeof value === "object" && value !== null) {
      return (
        <div key={fullKey} className="ml-2">
          <div className="font-mono font-bold">{key}:</div>
          {Object.entries(value).map(([k, v]) => renderYamlLine(k, v, fullKey))}
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
        <span className="font-bold">{key}:</span> {multiline && "|"}
        <span className="whitespace-pre-wrap">
          {typeof value === "string" && value.includes("\n")
            ? "\n" + value
            : String(value)}
        </span>
      </div>
    );
  };

  return (
    <div className="overflow-auto rounded border border-gray-200 bg-gray-50 p-4 text-xs">
      {Object.entries(json).map(([key, value]) => renderYamlLine(key, value))}
    </div>
  );
};
