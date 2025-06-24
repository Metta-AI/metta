"use client";
import { FC, useCallback, useEffect, useMemo, useRef, useState } from "react";

import { usePanZoom } from "@/hooks/use-pan-and-zoom";
import { Drawer } from "@/lib/draw/Drawer";
import { drawGrid } from "@/lib/draw/drawGrid";
import { Cell, MettaGrid } from "@/lib/MettaGrid";

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

function useSpacePressed(): boolean {
  const [spacePressed, setSpacePressed] = useState(false);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === " ") {
        event.preventDefault();
        event.stopPropagation();
        setSpacePressed(true);
      }
    };
    const handleKeyUp = (event: KeyboardEvent) => {
      if (event.key === " ") {
        event.preventDefault();
        event.stopPropagation();
        setSpacePressed(false);
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    document.addEventListener("keyup", handleKeyUp);

    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      document.removeEventListener("keyup", handleKeyUp);
    };
  }, []);

  return spacePressed;
}

const DebugInfo: FC<{
  canvas: HTMLCanvasElement | null;
  pan: { x: number; y: number };
  zoom: number;
}> = ({ canvas, pan, zoom }) => {
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
      <div>
        Pan: {round(pan.x)}x{round(pan.y)}
      </div>
      <div>Zoom: {round(zoom)}</div>
    </div>
  );
};

type CellHandler = (cell: Cell | undefined) => void;

type Props = {
  grid: MettaGrid;
  onCellHover?: CellHandler;
  selectedCell?: Cell;
  onCellSelect?: CellHandler;
  panOnSpace?: boolean;
};

export const MapViewer: FC<Props> = ({
  grid,
  onCellHover,
  onCellSelect,
  selectedCell,
  panOnSpace,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const dpr = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1;

  const spacePressed = useSpacePressed();

  const { setContainer, panZoomHandlers, setZoom, setPan, pan, zoom } =
    usePanZoom({
      minZoom: 1,
      maxZoom: 10,
      zoomSensitivity: 0.007,
      enablePan: !panOnSpace || spacePressed,
    });

  const drawer = useDrawer();

  const [hoveredCell, setHoveredCell] = useState<Cell | undefined>();

  const [scale, setScale] = useState(0);

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
  }, [grid.width, grid.height]);

  const recenter = useCallback(() => {
    if (!scale || !canvasRef.current) return;
    const canvas = canvasRef.current;
    setPan({
      x: (canvas.width - grid.width * scale) / (2 * dpr),
      y: (canvas.height - grid.height * scale) / (2 * dpr),
    });
    setZoom(1);
  }, [scale, setPan, setZoom]);

  useEffect(() => {
    initCanvas();
    recenter(); // intentionally not in the dependency array
  }, [initCanvas]);

  const transform = useMemo(() => {
    if (!canvasRef.current) return new DOMMatrixReadOnly();
    const ox = canvasRef.current.width / 2;
    const oy = canvasRef.current.height / 2;
    return new DOMMatrixReadOnly()
      .translate(pan.x * dpr, pan.y * dpr)
      .translate(ox, oy)
      .scale(zoom)
      .translate(-ox, -oy)
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

    context.lineWidth = 0.03;

    if (hoveredCell && grid.cellInGrid(hoveredCell)) {
      context.strokeStyle = "red";
      context.strokeRect(hoveredCell.c, hoveredCell.r, 1, 1);
    }

    if (selectedCell && grid.cellInGrid(selectedCell)) {
      context.strokeStyle = "blue";
      context.strokeRect(selectedCell.c, selectedCell.r, 1, 1);
    }
  }, [drawer, grid, transform, hoveredCell, selectedCell]);

  useCallOnWindowResize(initCanvas);

  // TODO - avoid rendering if not visible
  useEffect(draw, [draw]);

  // Benchmark: uncomment to redraw 60 frames per second when the canvas is visible on screen
  // useStressTest(draw, canvasRef.current);

  const cellFromMouseEvent = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>): Cell | undefined => {
      const point = transform.inverse().transformPoint({
        x: e.nativeEvent.offsetX * dpr,
        y: e.nativeEvent.offsetY * dpr,
      });
      const cell = {
        r: Math.floor(point.y),
        c: Math.floor(point.x),
      };
      if (!grid.cellInGrid(cell)) {
        return undefined;
      }
      return cell;
    },
    [transform, grid]
  );

  const onMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const cell = cellFromMouseEvent(e);

      setHoveredCell(cell);
      onCellHover?.(cell);
    },
    [zoom, grid, onCellHover, cellFromMouseEvent]
  );

  const onMouseClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!onCellSelect) return;

      const cell = cellFromMouseEvent(e);

      if (cell && grid.object(cell)) {
        onCellSelect(cell);
      } else {
        onCellSelect(undefined);
      }
    },
    [zoom, grid, onCellSelect, cellFromMouseEvent]
  );

  return (
    <div
      className="relative flex h-screen max-h-[600px] w-full overflow-hidden"
      {...panZoomHandlers}
      ref={setContainer}
    >
      <canvas
        ref={canvasRef}
        onDoubleClick={recenter}
        // Canvas takes all available space.
        className="h-full w-full cursor-grab overflow-hidden"
        onMouseMove={onMouseMove}
        onMouseLeave={() => {
          setHoveredCell(undefined);
          onCellHover?.(undefined);
        }}
        onClick={onMouseClick}
      />
      {/* <DebugInfo canvas={canvasRef.current} pan={pan} zoom={zoom} /> */}
    </div>
  );
};
