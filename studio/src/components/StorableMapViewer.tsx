"use client";
import Link from "next/link";
import { useQueryState } from "nuqs";
import { FC, useCallback, useMemo, useState } from "react";

import { FilterItem, parseFilterParam } from "@/app/stored-maps/dir/params";
import { SceneTree, StorableMap } from "@/lib/api";
import { MettaGrid } from "@/lib/MettaGrid";

import { CopyToClipboardButton } from "./CopyToClipboardButton";
import { JsonAsYaml } from "./JsonAsYaml";
import { MapViewer } from "./MapViewer";
import { SceneTreeViewer } from "./SceneTreeViewer";
import { Tabs } from "./Tabs";

// YAML viewer with the ability to click lines to filter the map list
const FilterableFrontmatterViewer: FC<{
  frontmatter: Record<string, unknown>;
}> = ({ frontmatter }) => {
  const [filters, setFilters] = useQueryState(
    "filter",
    parseFilterParam.withOptions({ shallow: false })
  );

  // Function to check if a key-value pair is currently in the filters
  const isFiltered = (key: string, value: string) => {
    if (key === "height") {
      console.log({ filters, key, value });
    }
    return (
      filters?.some(
        (filter) => `config.${key}` === filter.key && filter.value === value
      ) || false
    );
  };

  // Function to handle clicking on a frontmatter line
  const handleSelectLine = (configKey: string, value: string) => {
    const key = `config.${configKey}`;
    const newFilter: FilterItem = { key, value };

    // Check if this filter already exists
    const filterExists =
      filters?.some((filter) => filter.key === key && filter.value === value) ||
      false;

    if (filterExists) {
      // Remove the filter if it already exists
      setFilters(
        filters?.filter(
          (filter) => !(filter.key === key && filter.value === value)
        ) || []
      );
    } else {
      // Add the new filter
      setFilters([...(filters || []), newFilter]);
    }
  };

  return (
    <JsonAsYaml
      json={frontmatter}
      isSelected={isFiltered}
      onSelectLine={handleSelectLine}
    />
  );
};

export const StorableMapViewer: FC<{
  map: StorableMap;
  url?: string;
  // in /stored-maps list interface, we allow filtering by frontmatter props (which works by updating the URL)
  filterable?: boolean;
}> = ({ map, url, filterable = false }) => {
  // Parse the frontmatter YAML
  const grid = useMemo(() => MettaGrid.fromAscii(map.data), [map.data]);

  const [selectedSceneTree, setSelectedSceneTree] = useState<
    SceneTree | undefined
  >();

  const drawExtra = useCallback(
    (context: CanvasRenderingContext2D) => {
      if (!selectedSceneTree) return;
      context.save();

      // Create a clipping path for the selected area
      const margin = 0.1;
      context.beginPath();
      context.rect(
        selectedSceneTree.area.x + margin,
        selectedSceneTree.area.y + margin,
        selectedSceneTree.area.width - 2 * margin,
        selectedSceneTree.area.height - 2 * margin
      );

      // Invert the clipping path to draw fog everywhere EXCEPT the selected area
      context.rect(0, 0, grid.width, grid.height);
      context.clip("evenodd");

      // Draw fog-of-war overlay
      context.fillStyle = "rgba(0, 0, 0, 0.4)";
      context.fillRect(0, 0, grid.width, grid.height);

      context.restore();
    },
    [selectedSceneTree, grid]
  );

  return (
    <div className="grid min-h-[600px] grid-cols-[400px_1fr_250px] gap-8">
      <div className="max-h-[80vh] overflow-auto">
        <Tabs
          tabs={[
            {
              id: "config",
              label: "Config",
              content: filterable ? (
                <FilterableFrontmatterViewer
                  frontmatter={map.frontmatter.config}
                />
              ) : (
                <JsonAsYaml json={map.frontmatter.config} />
              ),
            },
            {
              id: "metadata",
              label: "Metadata",
              content: <JsonAsYaml json={map.frontmatter.metadata} />,
            },
            {
              id: "scene_tree",
              label: "Scene Tree",
              content: map.frontmatter.scene_tree ? (
                <SceneTreeViewer
                  sceneTree={map.frontmatter.scene_tree}
                  onSceneSelect={(sceneTree) => {
                    setSelectedSceneTree(sceneTree);
                  }}
                />
              ) : (
                <div className="pt-4 text-center text-gray-500">
                  No scene tree
                </div>
              ),
            },
          ]}
          defaultTab="config"
        />
      </div>
      <div className="flex flex-col items-center justify-start overflow-auto">
        <MapViewer grid={grid} drawExtra={drawExtra} />
      </div>
      <div className="flex flex-col gap-2">
        <CopyToClipboardButton text={map.data}>
          Copy Map Data to Clipboard
        </CopyToClipboardButton>
        {url &&
          typeof window !== "undefined" &&
          window.location.pathname !== "/stored-maps/view" && (
            <Link
              className="text-blue-500 hover:underline"
              href={`/stored-maps/view?map=${url}`}
              target="_blank"
            >
              Permalink
            </Link>
          )}
      </div>
    </div>
  );
};
