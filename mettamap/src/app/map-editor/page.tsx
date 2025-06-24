"use client";
import { useMemo, useState } from "react";

import { Button } from "@/components/Button";
import { MapViewer } from "@/components/MapViewer";
import { NumberInput } from "@/components/NumberInput";
import { useIsMouseDown } from "@/hooks/useIsMouseDown";
import { Cell, MettaGrid } from "@/lib/MettaGrid";

import { AsciiPreview } from "./AsciiPreview";
import { ObjectsPanel } from "./ObjectsPanel";

export default function MapEditorPage() {
  const [gridWidth, setGridWidth] = useState(10);
  const [gridHeight, setGridHeight] = useState(10);

  const makeGrid = (w: number, h: number) => MettaGrid.empty(w, h);
  const [grid, setGrid] = useState<MettaGrid>(() =>
    makeGrid(gridWidth, gridHeight)
  );

  const [selectedEntity, setSelectedEntity] = useState("wall");
  const [confirmReset, setConfirmReset] = useState(false);

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
    <div>
      <div className="flex items-end gap-1 p-2">
        <label>
          <span className="text-sm">Width:</span>
          <NumberInput
            value={gridWidth}
            onChange={(e) => setGridWidth(+e.target.value)}
          />
        </label>
        <label>
          <span className="text-sm">Height:</span>
          <NumberInput
            value={gridHeight}
            onChange={(e) => setGridHeight(+e.target.value)}
          />
        </label>
        <Button
          onClick={() => {
            if (!confirmReset) {
              setConfirmReset(true);
              setTimeout(() => setConfirmReset(false), 5000);
            } else {
              setGrid(makeGrid(gridWidth, gridHeight));
              setConfirmReset(false);
            }
          }}
        >
          {confirmReset ? "Are you sure?" : "Create Grid"}
        </Button>
        <Button
          onClick={() => {
            navigator.clipboard.writeText(asciiPreview);
          }}
        >
          Copy ASCII
        </Button>
      </div>
      <div className="p-2">
        <ObjectsPanel
          selectedEntity={selectedEntity}
          setSelectedEntity={setSelectedEntity}
        />
      </div>
      <div className="max-h-screen">
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
      <AsciiPreview ascii={asciiPreview} />
    </div>
  );
}
