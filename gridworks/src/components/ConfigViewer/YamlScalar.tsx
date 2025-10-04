"use client";
import { FC, ReactNode, use } from "react";

import { RepoRootContext } from "../global-contexts/RepoRootContext";
import { ConfigNode } from "./utils";

export const YamlScalar: FC<{
  node: ConfigNode<string | number | boolean | null>;
}> = ({ node }) => {
  const value = node.value;
  if (value === null) {
    return <span className="text-gray-500">null</span>;
  }

  const multiline = typeof value === "string" && value.includes("\n");

  let url = "";
  const KNOWN_PACKAGES = [
    "mettagrid.mapgen",
    "mettagrid.map_builder",
    "metta.map",
    "metta.cogworks",
    "experiments.recipes",
  ];

  const repoRoot = use(RepoRootContext);
  if (
    typeof value === "string" &&
    KNOWN_PACKAGES.some((p) => value.startsWith(p))
  ) {
    // looks like a Python class name - mettagrid.mapgen.scenes.random.Random
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
