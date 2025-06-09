"use client";
import { FC, useCallback, useEffect, useRef, useState } from "react";

import { usePanZoom } from "@/hooks/use-pan-and-zoom";
import { Drawer } from "@/lib/draw/Drawer";
import { drawGrid } from "@/lib/draw/drawGrid";
import { Cell, MettaGrid } from "@/lib/MettaGrid";

type CellHandler = (cell: Cell | undefined) => void;

type Props = {
  grid: MettaGrid;
  onCellHover?: CellHandler;
  selectedCell?: Cell;
  onCellSelect?: CellHandler;
};

function useDrawer(): Drawer | undefined {
  const [drawer, setDrawer] = useState<Drawer | undefined>();
  useEffect(() => {
    Drawer.load().then(setDrawer);
  }, []);

  return drawer;
}

const Overlay: FC<{
  cellSize: number;
  hoveredCell?: Cell;
  selectedCell?: Cell;
}> = ({ cellSize, hoveredCell, selectedCell }) => {
  return (
    <>
      {hoveredCell && (
        <div
          className="absolute border border-blue-500"
          style={{
            left: hoveredCell?.c * cellSize,
            top: hoveredCell?.r * cellSize,
            width: cellSize + 2,
            height: cellSize + 2,
          }}
        ></div>
      )}
      {selectedCell && (
        <div
          className="absolute border border-red-500"
          style={{
            left: selectedCell?.c * cellSize,
            top: selectedCell?.r * cellSize,
            width: cellSize + 2,
            height: cellSize + 2,
          }}
        ></div>
      )}
    </>
  );
};

export const MapViewer: FC<Props> = ({
  grid,
  onCellHover,
  onCellSelect,
  selectedCell,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const { transform, setContainer, panZoomHandlers, setZoom, setPan, zoom } =
    usePanZoom({
      minZoom: 1,
      maxZoom: 10,
      zoomSensitivity: 0.007,
    });

  const drawer = useDrawer();

  // Cell size used for drawing the grid.
  // This is in internal canvas pixels, not pixels on the screen. (canvas.width, not clientWidth)
  const [cellSize, setCellSize] = useState(0);

  const [hoveredCell, setHoveredCell] = useState<Cell | undefined>();

  const measureCellSize = useCallback(() => {
    if (!canvasRef.current || !containerRef.current) return;

    const containerWidth = containerRef.current.clientWidth;
    const containerHeight = containerRef.current.clientHeight;

    // Calculate new cell size
    const widthBasedSize = Math.floor(containerWidth / grid.width);
    const heightBasedSize = Math.floor(containerHeight / grid.height);

    // Large minimal cell size is useful for zoom, but not very effective, could be optimized.
    // This results in 3k * 3k canvas for 120x120 grid.
    // (e.g. with Factorio-style discrete zoom and pre-rendered sprites for each size)
    const cellSize = Math.max(24, Math.min(widthBasedSize, heightBasedSize));
    setCellSize(cellSize);

    // Set canvas dimensions
    canvasRef.current.width = grid.width * cellSize;
    canvasRef.current.height = grid.height * cellSize;
  }, [grid]);

  useEffect(() => {
    measureCellSize();
  }, [measureCellSize]);

  const draw = useCallback(() => {
    if (!drawer || !canvasRef.current || !cellSize) return;

    drawGrid({
      grid,
      canvas: canvasRef.current,
      drawer: drawer,
      cellSize,
    });
  }, [drawer, grid, cellSize]);

  // Handle window resize
  useEffect(() => {
    // TODO - avoid rendering? (doesn't work yet)
    window.addEventListener("resize", draw);
    return () => window.removeEventListener("resize", draw);
  }, [draw]);

  // TODO - avoid rendering if not visible
  useEffect(draw, [draw]);

  // Benchmark: uncomment to redraw 60 frames per second when the canvas is visible on screen
  // useStressTest(draw, canvasRef.current);

  const cellFromMouseEvent = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>): Cell | null => {
      if (!canvasRef.current) return null;

      // 1. Grab the bounding box AFTER CSS transforms:
      const rect = canvasRef.current.getBoundingClientRect();

      // 2. Compute screen‑relative coords inside that box
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;

      const c = sx * (grid.width / rect.width);
      const r = sy * (grid.height / rect.height);

      return { r: Math.floor(r), c: Math.floor(c) };
    },
    [grid]
  );

  const onMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const cell = cellFromMouseEvent(e);
      if (!cell) return;

      setHoveredCell(cell);
      onCellHover?.(cell);
    },
    [zoom, grid, onCellHover, cellFromMouseEvent]
  );

  const onMouseClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!onCellSelect) return;

      const cell = cellFromMouseEvent(e);
      if (!cell) return;

      if (grid.object(cell)) {
        onCellSelect(cell);
      } else {
        onCellSelect(undefined);
      }
    },
    [zoom, grid, onCellSelect, cellFromMouseEvent]
  );

  return (
    <div
      ref={(el) => {
        containerRef.current = el;
        setContainer(el);
      }}
      {...panZoomHandlers}
      onDoubleClick={() => {
        // Reset on double click
        setZoom(1);
        setPan({ x: 0, y: 0 });
      }}
      className="flex h-full w-full cursor-grab items-start justify-center overflow-hidden bg-gray-100"
    >
      <div className="relative max-h-full max-w-full" style={{ transform }}>
        <canvas
          ref={canvasRef}
          onMouseMove={onMouseMove}
          onMouseLeave={() => {
            setHoveredCell(undefined);
            onCellHover?.(undefined);
          }}
          onClick={onMouseClick}
          className="h-full max-h-full w-full max-w-full"
        />
        <div className="pointer-events-none absolute inset-0 z-10">
          {canvasRef.current && (
            <Overlay
              cellSize={canvasRef.current?.clientWidth / grid.width}
              hoveredCell={hoveredCell}
              selectedCell={selectedCell}
            />
          )}
        </div>
      </div>
    </div>
  );
};
