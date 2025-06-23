"use client";
import { FC, useCallback, useEffect, useMemo, useRef, useState } from "react";

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

function useCallOnWindowResize(callback: () => void) {
  useEffect(() => {
    window.addEventListener("resize", callback);
    return () => window.removeEventListener("resize", callback);
  }, [callback]);
}

const DebugInfo: FC<{ canvas: HTMLCanvasElement | null }> = ({ canvas }) => {
  if (!canvas) return null;

  const round = (x: number) => Math.round(x * 100) / 100;

  return (
    <div className="absolute right-0 bottom-0 z-10 text-xs">
      <div>
        Canvas size: {canvas.width}x{canvas.height}
      </div>
      <div>
        Client size: {canvas.clientWidth}x{canvas.clientHeight}
      </div>
      <div>
        Bounding rect: {round(canvas.getBoundingClientRect().width)}x
        {round(canvas.getBoundingClientRect().height)}
      </div>
    </div>
  );
};

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

  const dpr = window.devicePixelRatio || 1;

  const { setContainer, panZoomHandlers, setZoom, setPan, pan, zoom } =
    usePanZoom({
      minZoom: 1,
      maxZoom: 10,
      zoomSensitivity: 0.007,
    });

  const drawer = useDrawer();

  const [hoveredCell, setHoveredCell] = useState<Cell | undefined>();

  const [scale, setScale] = useState(0);

  const initCanvasRef = useCallback(
    (el: HTMLCanvasElement | null) => {
      canvasRef.current = el;
      setContainer(el);
    },
    [setContainer]
  );

  // measure cell size and set canvas dimensions
  const initCanvas = useCallback(() => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;

    // Set canvas dimensions
    canvas.width = canvas.clientWidth * dpr;
    canvas.height = canvas.clientHeight * dpr;

    // Calculate new scale
    const widthBasedSize = canvas.width / grid.width;
    const heightBasedSize = canvas.height / grid.height;

    setScale(Math.min(widthBasedSize, heightBasedSize));
  }, [grid]);

  useEffect(() => {
    initCanvas();
  }, [initCanvas]);

  const transform = useMemo(() => {
    // Apply translation before scaling so that the pan values are not affected by the zoom factor.
    // This keeps the point under the cursor fixed while zooming.
    return new DOMMatrixReadOnly()
      .translate(pan.x * dpr, pan.y * dpr)
      .scale(zoom)
      .scale(scale);
  }, [zoom, pan, scale, dpr]);

  const draw = useCallback(() => {
    if (!drawer || !canvasRef.current) return;

    const canvas = canvasRef.current;

    const context = canvas.getContext("2d");
    if (!context) return;

    context.resetTransform();
    context.fillStyle = "#eee";
    context.fillRect(0, 0, canvas.width, canvas.height);

    context.setTransform(transform);

    drawGrid({
      grid,
      context,
      drawer,
    });
  }, [drawer, grid, transform]);

  useCallOnWindowResize(initCanvas);

  // TODO - avoid rendering if not visible
  useEffect(draw, [draw]);

  // Benchmark: uncomment to redraw 60 frames per second when the canvas is visible on screen
  // useStressTest(draw, canvasRef.current);

  const cellFromMouseEvent = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>): Cell | null => {
      const point = transform.inverse().transformPoint({
        x: e.nativeEvent.offsetX * dpr,
        y: e.nativeEvent.offsetY * dpr,
      });
      return {
        r: Math.floor(point.y),
        c: Math.floor(point.x),
      };
    },
    [transform, grid]
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
    <div className="relative flex h-screen w-full overflow-hidden">
      <canvas
        ref={initCanvasRef}
        {...panZoomHandlers}
        onDoubleClick={() => {
          // Reset on double click
          setZoom(1);
          setPan({ x: 0, y: 0 });
        }}
        // Canvas takes all available space.
        className="h-full w-full cursor-grab overflow-hidden"
        // ref={canvasRef}
        // onMouseMove={onMouseMove}
        // onMouseLeave={() => {
        //   setHoveredCell(undefined);
        //   onCellHover?.(undefined);
        // }}
        onClick={onMouseClick}
        // className="h-full max-h-full w-full max-w-full"
      />
      <DebugInfo canvas={canvasRef.current} />
      {canvasRef.current && (
        <div className="pointer-events-none absolute inset-0 z-10">
          <Overlay
            cellSize={canvasRef.current?.clientWidth / grid.width}
            hoveredCell={hoveredCell}
            selectedCell={selectedCell}
          />
        </div>
      )}
    </div>
  );
};
