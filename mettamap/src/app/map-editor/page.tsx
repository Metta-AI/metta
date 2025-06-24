"use client";
import { useMemo, useState } from "react";

import { Button } from "@/components/Button";
import { MapViewer } from "@/components/MapViewer";
import { Tabs } from "@/components/Tabs";
import { useIsMouseDown } from "@/hooks/useIsMouseDown";
import { Cell, MettaGrid } from "@/lib/MettaGrid";

import { AsciiPreview } from "./AsciiPreview";
import { ObjectsPanel } from "./ObjectsPanel";
import { ResetGridButton } from "./ResetGridButton";

export default function MapEditorPage() {
  const [grid, setGrid] = useState<MettaGrid>(() => MettaGrid.empty(10, 10));

  const [selectedEntity, setSelectedEntity] = useState("wall");
  const drawCell = (cell: Cell | undefined) => {
    if (!cell) return;
    setGrid((g) => {
      const newG = g.replaceCellByName(cell.r, cell.c, selectedEntity);
      return newG;
    });
  };

  const asciiPreview = useMemo(() => grid.toAscii(), [grid]);

  const isMouseDown = useIsMouseDown();

  return (
    <div className="flex h-screen">
      {/* Left Sidebar */}
      <div className="w-64 border-r border-gray-300 bg-gray-50 p-4">
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
                  <AsciiPreview ascii={asciiPreview} />
                </div>
              </div>
            ),
          },
        ]}
        defaultTab="map"
        additionalTabBarContent={
          <Button
            onClick={() => {
              navigator.clipboard.writeText(asciiPreview);
            }}
            theme="primary"
          >
            Copy ASCII
          </Button>
        }
      />
    </div>
  );
}
