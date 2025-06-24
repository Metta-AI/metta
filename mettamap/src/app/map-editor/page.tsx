"use client";
import clsx from "clsx";
import { FC, useMemo, useState } from "react";

import { Button } from "@/components/Button";
import { MapViewer } from "@/components/MapViewer";
import { Modal } from "@/components/Modal";
import { NumberInput } from "@/components/NumberInput";
import { useIsMouseDown } from "@/hooks/useIsMouseDown";
import { Cell, MettaGrid } from "@/lib/MettaGrid";

import { AsciiPreview } from "./AsciiPreview";
import { ObjectsPanel } from "./ObjectsPanel";

const ResetGridModal: FC<{
  isOpen: boolean;
  onClose: () => void;
  onCreateGrid: (width: number, height: number) => void;
  currentWidth: number;
  currentHeight: number;
}> = ({ isOpen, onClose, onCreateGrid, currentWidth, currentHeight }) => {
  const [width, setWidth] = useState(currentWidth);
  const [height, setHeight] = useState(currentHeight);

  if (!isOpen) return null;

  return (
    <Modal onClose={onClose}>
      <Modal.Header>Reset Grid</Modal.Header>
      <Modal.Body>
        <label>
          <span className="mb-1 block text-sm">Width:</span>
          <NumberInput
            value={width}
            onChange={(e) => setWidth(+e.target.value)}
          />
        </label>
        <label>
          <span className="mb-1 block text-sm">Height:</span>
          <NumberInput
            value={height}
            onChange={(e) => setHeight(+e.target.value)}
          />
        </label>
      </Modal.Body>
      <Modal.Footer>
        <Button onClick={onClose}>Cancel</Button>
        <Button
          onClick={() => {
            onCreateGrid(width, height);
            onClose();
          }}
        >
          Create Grid
        </Button>
      </Modal.Footer>
    </Modal>
  );
};

export default function MapEditorPage() {
  const [gridWidth, setGridWidth] = useState(10);
  const [gridHeight, setGridHeight] = useState(10);

  const makeGrid = (w: number, h: number) => MettaGrid.empty(w, h);
  const [grid, setGrid] = useState<MettaGrid>(() =>
    makeGrid(gridWidth, gridHeight)
  );

  const [selectedEntity, setSelectedEntity] = useState("wall");
  const [showResetModal, setShowResetModal] = useState(false);
  const [activeTab, setActiveTab] = useState<"map" | "ascii">("map");

  const drawCell = (cell: Cell | undefined) => {
    if (!cell) return;
    setGrid((g) => {
      const newG = g.replaceCellByName(cell.r, cell.c, selectedEntity);
      return newG;
    });
  };

  const asciiPreview = useMemo(() => grid.toAscii(), [grid]);

  const isMouseDown = useIsMouseDown();

  const handleCreateGrid = (width: number, height: number) => {
    setGridWidth(width);
    setGridHeight(height);
    setGrid(makeGrid(width, height));
  };

  return (
    <div className="flex h-screen">
      {/* Left Sidebar */}
      <div className="w-64 border-r border-gray-300 bg-gray-50 p-4">
        <div className="mb-4">
          <h2 className="mb-2 text-lg font-semibold">Tools</h2>
          <Button onClick={() => setShowResetModal(true)}>Reset Grid</Button>
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
      <div className="flex flex-1 flex-col">
        {/* Tabs */}
        <div className="flex border-b border-gray-300">
          <button
            onClick={() => setActiveTab("map")}
            className={clsx(
              "cursor-pointer px-4 py-2 font-medium",
              activeTab === "map"
                ? "border-b-2 border-blue-600 text-blue-600"
                : "text-gray-500 hover:text-gray-900"
            )}
          >
            Map Editor
          </button>
          <button
            onClick={() => setActiveTab("ascii")}
            className={clsx(
              "cursor-pointer px-4 py-2 font-medium",
              activeTab === "ascii"
                ? "border-b-2 border-blue-600 text-blue-600"
                : "text-gray-500 hover:text-gray-900"
            )}
          >
            ASCII Preview
          </button>
          <div className="ml-4 flex items-center">
            <Button
              onClick={() => {
                navigator.clipboard.writeText(asciiPreview);
              }}
              theme="primary"
            >
              Copy ASCII
            </Button>
          </div>
        </div>

        <div className="flex-1 overflow-hidden">
          {activeTab === "map" && (
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
          )}

          {activeTab === "ascii" && (
            <div className="flex h-full flex-col">
              <div className="flex-1 overflow-auto p-4">
                <AsciiPreview ascii={asciiPreview} />
              </div>
            </div>
          )}
        </div>
      </div>

      <ResetGridModal
        isOpen={showResetModal}
        onClose={() => setShowResetModal(false)}
        onCreateGrid={handleCreateGrid}
        currentWidth={gridWidth}
        currentHeight={gridHeight}
      />
    </div>
  );
}
