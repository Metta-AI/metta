"use client";
import { useState } from "react";

import { Button } from "@/components/Button";
import { MapViewer } from "@/components/MapViewer";
import { useSpacePressed } from "@/components/MapViewer/hooks";
import { Tabs } from "@/components/Tabs";
import { useIsMouseDown } from "@/hooks/useIsMouseDown";
import { Cell, MettaGrid } from "@/lib/MettaGrid";

import { AsciiEditor } from "./AsciiEditor";
import { ObjectsPanel } from "./ObjectsPanel";
import { ResetGridButton } from "./ResetGridButton";

export default function MapEditorPage() {
  const [grid, setGrid] = useState<MettaGrid>(() => MettaGrid.empty(10, 10));

  const [selectedEntity, setSelectedEntity] = useState("wall");

  const isMouseDown = useIsMouseDown();
  const isSpacePressed = useSpacePressed();

  const drawCell = (cell: Cell | undefined) => {
    if (!cell || isSpacePressed) return;
    setGrid((g) => {
      const newG = g.replaceCellByName(cell.r, cell.c, selectedEntity);
      return newG;
    });
  };

  return (
    <div className="flex h-screen">
      {/* Left Sidebar */}
      <div className="w-80 border-r border-gray-300 bg-gray-50 p-4">
        <div className="mb-4">
          <h2 className="mb-2 text-lg font-semibold">Tools</h2>
          <ResetGridButton currentGrid={grid} setGrid={setGrid} />
        </div>

        <div>
          <h3 className="mb-2 text-sm font-semibold">Objects</h3>
          <ObjectsPanel
            selectedEntity={selectedEntity}
            setSelectedEntity={setSelectedEntity}
          />
        </div>
      </div>

      {/* Main Content Area */}
      <Tabs
        tabs={[
          {
            id: "map",
            label: "Map Editor",
            content: (
              <div className="h-full">
                <MapViewer
                  grid={grid}
                  onCellSelect={drawCell}
                  onCellHover={(cell) => {
                    if (isMouseDown) {
                      drawCell(cell);
                    }
                  }}
                  panOnSpace
                />
              </div>
            ),
          },
          {
            id: "ascii",
            label: "ASCII Preview",
            content: (
              <div className="flex h-full flex-col">
                <div className="flex-1 overflow-auto p-4">
                  <AsciiEditor grid={grid} setGrid={setGrid} />
                </div>
              </div>
            ),
          },
        ]}
        defaultTab="map"
        additionalTabBarContent={
          <div className="flex items-center gap-4">
            <Button
              onClick={() => {
                navigator.clipboard.writeText(grid.toAscii());
              }}
              theme="primary"
              size="sm"
            >
              Copy ASCII
            </Button>
            <div className="text-xs text-gray-700">
              {
                'Tip: hold down "Space" to pan. Double-click to reset zoom and pan.'
              }
            </div>
          </div>
        }
      />
    </div>
  );
}
