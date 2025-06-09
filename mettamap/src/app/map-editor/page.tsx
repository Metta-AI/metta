"use client";
import clsx from "clsx";
import { FC, useEffect, useMemo, useState } from "react";
import TextareaAutosize from "react-textarea-autosize";

import { Button } from "@/components/Button";
import { MapViewer } from "@/components/MapViewer";
import { NumberInput } from "@/components/NumberInput";
import { loadMettaTileSets } from "@/lib/draw/mettaTileSets";
import { TileSetCollection } from "@/lib/draw/TileSet";
import { MettaGrid, ObjectName } from "@/lib/MettaGrid";

const AsciiPreview: FC<{ ascii: string }> = ({ ascii }) => {
  return (
    <TextareaAutosize readOnly value={ascii} className="w-full font-mono" />
  );
};

const teamColors = {
  "agent.team_1": "#d9534f",
  "agent.team_2": "#0275d8",
  "agent.team_3": "#5cb85c",
  "agent.team_4": "#f0ad4e",
};

export const ObjectsPanel: FC<{
  selectedEntity: ObjectName;
  setSelectedEntity: (entity: ObjectName) => void;
  cellSize: number;
}> = ({ selectedEntity, setSelectedEntity, cellSize }) => {
  const [tileSets, setTileSets] = useState<TileSetCollection | null>(null);
  useEffect(() => {
    loadMettaTileSets().then(setTileSets);
  }, []);

  if (!tileSets) {
    return null;
  }

  return (
    <div className="flex gap-1">
      {Object.keys(tileSets.nameToTileSet).map((key) => {
        const { wrapper, inner } = tileSets.css(key, cellSize);
        return (
          <button
            key={key}
            onClick={() => setSelectedEntity(key as ObjectName)}
            className={clsx(
              "cursor-pointer",
              selectedEntity === key
                ? "ring-2 ring-blue-500"
                : "hover:ring-2 hover:ring-blue-300"
            )}
            title={key}
          >
            <div style={wrapper}>
              <div style={inner} />
            </div>
          </button>
        );
      })}
    </div>
  );
};

export default function MapEditorPage() {
  const cellSize = 20;

  // Inputs
  const [gridWidth, setGridWidth] = useState(10);
  const [gridHeight, setGridHeight] = useState(10);

  const makeGrid = (w: number, h: number) => MettaGrid.empty(w, h);
  const [grid, setGrid] = useState<MettaGrid>(() =>
    makeGrid(gridWidth, gridHeight)
  );

  const [selectedEntity, setSelectedEntity] = useState<ObjectName>("wall");
  const [asciiEditMode, setAsciiEditMode] = useState(false);
  const [confirmReset, setConfirmReset] = useState(false);

  const drawCell = (cell: { x: number; y: number } | undefined) => {
    if (!cell) return;
    setGrid((g) => {
      const newG = g.replaceCellByName(cell.y, cell.x, selectedEntity);
      return newG;
    });
  };

  const asciiPreview = useMemo(() => grid.toAscii(), [grid]);

  return (
    <div className="map-editor">
      <div className="flex items-end gap-1">
        <label>
          Width:
          <NumberInput
            value={gridWidth}
            onChange={(e) => setGridWidth(+e.target.value)}
          />
        </label>
        <label>
          Height:
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
          cellSize={cellSize}
        />
      </div>
      <div className="max-h-screen">
        <MapViewer grid={grid} onCellSelect={drawCell} />
      </div>
      <AsciiPreview ascii={asciiPreview} />
    </div>
  );
}
