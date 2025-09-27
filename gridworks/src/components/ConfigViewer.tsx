"use client";
import clsx from "clsx";
import { createContext, FC, ReactNode, use } from "react";

import { RepoRootContext } from "./RepoRootContext";

const YamlContext = createContext<{
  isSelected?: (key: string, value: string) => boolean;
  onSelectLine?: (key: string, value: string) => void;
  unsetFields: Set<string>;
}>({ unsetFields: new Set() });

const YamlKey: FC<{
  name: string;
  disabled?: boolean;
}> = ({ name, disabled }) => {
  return (
    <span
      className={clsx(
        disabled ? "text-gray-500" : "font-semibold text-blue-900"
      )}
    >
      {name}:
    </span>
  );
};

const YamlScalar: FC<{
  value: string | number | boolean | null;
}> = ({ value }) => {
  if (value === null) {
    return <span className="text-gray-500">null</span>;
  }

  const multiline = typeof value === "string" && value.includes("\n");

  let url = "";
  const KNOWN_PACKAGES = [
    "mettagrid.mapgen",
    "mettagrid.map_builder",
    "metta.map",
    "softmax.cogworks",
  ];

  const repoRoot = use(RepoRootContext);
  if (
    typeof value === "string" &&
    KNOWN_PACKAGES.some((p) => value.startsWith(p))
  ) {
    // looks like a Hydra target - mod1.mod2.ClassName
    const parts = value.split(".");
    parts.pop();
    let filename = parts.join("/") + ".py";
    if (value.startsWith("mettagrid")) {
      filename = `packages/mettagrid/python/src/${filename}`;
    }
    url = `cursor://file${repoRoot}/${filename}`;
  }

  if (
    typeof value === "string" &&
    value.startsWith("packages/mettagrid/configs/maps/") &&
    value.endsWith(".map")
  ) {
    url = `cursor://file${repoRoot}/${value}`;
  }

  let valueEl: ReactNode =
    typeof value === "string" && value.includes("\n")
      ? "\n" + value
      : String(value);

  if (url) {
    valueEl = (
      <a href={url} className="text-blue-800 hover:underline">
        {valueEl}
      </a>
    );
  }

  return (
    <div>
      {multiline && "|"}
      <span className="whitespace-pre-wrap">{valueEl}</span>
    </div>
  );
};

function isScalar(value: unknown): value is string | number | boolean | null {
  return (
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean" ||
    value === null
  );
}

const YamlKeyValue: FC<{
  yamlKey: string;
  value: unknown;
  path: string;
  depth: number;
}> = ({ yamlKey, value, path, depth }) => {
  const fullKey = path ? `${path}.${yamlKey}` : yamlKey;

  if (isScalar(value)) {
    const { isSelected, onSelectLine, unsetFields } = use(YamlContext);

    const isActive = isSelected?.(fullKey, String(value));

    const onClick = onSelectLine
      ? () => onSelectLine(fullKey, String(value))
      : undefined;

    return (
      <div
        className={clsx(
          "flex gap-1",
          isActive && "-mx-1 bg-blue-100 px-1",
          onClick && "cursor-pointer hover:bg-blue-200"
        )}
        onClick={onClick}
      >
        <YamlKey name={yamlKey} disabled={unsetFields.has(fullKey)} />
        <YamlScalar value={value} />
      </div>
    );
  }

  const singleLine =
    (Array.isArray(value) && value.length === 0) ||
    (typeof value === "object" && Object.keys(value).length === 0);

  return (
    <div className={clsx(singleLine && "flex gap-1")}>
      <YamlKey name={yamlKey} />
      <div className={clsx(!singleLine && "ml-[2ch]")}>
        <YamlAny value={value} path={fullKey} depth={depth + 1} />
      </div>
    </div>
  );
};

const YamlObject: FC<{
  value: object;
  path: string;
  depth: number;
}> = ({ value, path, depth }) => {
  if (Object.keys(value).length === 0) {
    return <div className="text-gray-500">{"{}"}</div>;
  }

  return (
    <div>
      {Object.entries(value).map(([key, value]) => (
        <YamlKeyValue
          key={key}
          yamlKey={key}
          value={value}
          path={path}
          depth={depth}
        />
      ))}
    </div>
  );
};

const YamlArray: FC<{
  value: unknown[];
  path: string;
  depth: number;
}> = ({ value, path, depth }) => {
  if (value.length === 0) {
    return <div className="text-gray-500">[]</div>;
  }

  if (
    path.includes(".map_data") &&
    value.every((v) => typeof v === "string" && v.length === 1)
  ) {
    // special case for readability
    return <div>{value.join("")}</div>;
  }

  return (
    <div className="-ml-[2ch]">
      {value.map((v, i) => (
        <div key={`${path}.${i}`} className="flex gap-[1ch]">
          <div className="font-semibold text-blue-900">-</div>
          <YamlAny value={v} path={path} depth={depth + 1} />
        </div>
      ))}
    </div>
  );
};

const YamlAny: FC<{
  value: unknown;
  path: string;
  depth: number;
}> = ({ value, path, depth }) => {
  if (Array.isArray(value)) {
    return <YamlArray value={value} path={path} depth={depth} />;
  }
  if (typeof value === "object" && value !== null) {
    return <YamlObject value={value} path={path} depth={depth} />;
  }

  if (isScalar(value)) {
    return <YamlScalar value={value} />;
  }

  throw new Error(`Unknown value type: ${typeof value}`);
};

export const ConfigViewer: FC<{
  value: unknown;
  // The fields that weren't set explicitly (so they can still have their default values, not necessarily null).
  // The prop name mirrors `model_fields_set` in Pydantic.
  unsetFields?: string[];
  isSelected?: (key: string, value: string) => boolean;
  onSelectLine?: (key: string, value: string) => void;
}> = ({ value, isSelected, onSelectLine, unsetFields }) => {
  return (
    <YamlContext.Provider
      value={{
        isSelected,
        onSelectLine,
        unsetFields: new Set(unsetFields ?? []),
      }}
    >
      <div className="overflow-auto rounded border border-gray-200 bg-gray-50 p-4 font-mono text-xs">
        <YamlAny value={value} path="" depth={0} />
      </div>
    </YamlContext.Provider>
  );
};
