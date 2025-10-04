"use client";
import { FC, useCallback, useMemo, useState } from "react";

import { SceneTree, StorableMap } from "@/lib/api";
import { MettaGrid } from "@/lib/MettaGrid";

import { ConfigViewer } from "./ConfigViewer";
import { CopyToClipboardButton } from "./CopyToClipboardButton";
import { MapViewer } from "./MapViewer";
import { SceneTreeViewer } from "./SceneTreeViewer";
import { Tabs } from "./Tabs";

export const StorableMapViewer: FC<{
  map: StorableMap;
}> = ({ map }) => {
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
              content: <ConfigViewer value={map.frontmatter.config} />,
            },
            {
              id: "metadata",
              label: "Metadata",
              content: <ConfigViewer value={map.frontmatter.metadata} />,
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
      </div>
    </div>
  );
};
