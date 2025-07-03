"use client";
import clsx from "clsx";
import { createContext, FC, ReactNode, use } from "react";

import { RepoRootContext } from "./RepoRootContext";

const YamlContext = createContext<{
  isSelected?: (key: string, value: string) => boolean;
  onSelectLine?: (key: string, value: string) => void;
}>({});

const YamlKey: FC<{
  name: string;
}> = ({ name }) => {
  return <span className="font-semibold text-blue-900">{name}:</span>;
};

const YamlScalar: FC<{
  value: string | number | boolean;
}> = ({ value }) => {
  const multiline = typeof value === "string" && value.includes("\n");

  let url = "";
  const KNOWN_PACKAGES = [
    "metta.map",
    "metta.mettagrid.room",
    "metta.mettagrid.curriculum",
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
    if (value.startsWith("metta.mettagrid")) {
      filename = `mettagrid/src/${filename}`;
    }
    url = `cursor://file${repoRoot}/${filename}`;
  }

  if (
    typeof value === "string" &&
    value.startsWith("configs/env/mettagrid/maps/") &&
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

function isScalar(value: unknown): value is string | number | boolean {
  return (
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
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
    const { isSelected, onSelectLine } = use(YamlContext);

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
        <YamlKey name={yamlKey} />
        <YamlScalar value={value} />
      </div>
    );
  }

  return (
    <div>
      <YamlKey name={yamlKey} />
      <div className="ml-[2ch]">
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

  if (
    typeof value === "boolean" ||
    typeof value === "number" ||
    typeof value === "string"
  ) {
    return <YamlScalar value={value} />;
  }

  if (value === null) {
    return <span className="text-gray-500">null</span>;
  }

  throw new Error(`Unknown value type: ${typeof value}`);
};

export const JsonAsYaml: FC<{
  json: Record<string, unknown>;
  isSelected?: (key: string, value: string) => boolean;
  onSelectLine?: (key: string, value: string) => void;
}> = ({ json, isSelected, onSelectLine }) => {
  return (
    <YamlContext.Provider value={{ isSelected, onSelectLine }}>
      <div className="overflow-auto rounded border border-gray-200 bg-gray-50 p-4 font-mono text-xs">
        <YamlAny value={json} path="" depth={0} />
      </div>
    </YamlContext.Provider>
  );
};
