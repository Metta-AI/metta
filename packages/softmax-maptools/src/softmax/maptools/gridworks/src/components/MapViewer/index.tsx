"use client";
import clsx from "clsx";
import { FC, useCallback, useEffect, useMemo, useRef, useState } from "react";

import { usePanZoom } from "@/hooks/use-pan-and-zoom";
import { useIsMouseDown } from "@/hooks/useIsMouseDown";
import { drawGrid } from "@/lib/draw/drawGrid";
import { Cell, MettaGrid } from "@/lib/MettaGrid";

import {
  useCallOnElementResize,
  useCallOnWindowResize,
  useDrawer,
  useSpacePressed,
} from "./hooks";

type CellHandler = (cell: Cell | undefined) => void;

// Useful for external components to draw on top of the map, e.g. for drawing areas from the scene tree.
export type MapDrawer = (context: CanvasRenderingContext2D) => void;

type Props = {
  grid: MettaGrid;
  onCellHover?: CellHandler;
  selectedCell?: Cell;
  onCellSelect?: CellHandler;
  panOnSpace?: boolean;
  drawExtra?: MapDrawer;
};

const MapViewerBrowserOnly: FC<Props> = ({
  grid,
  onCellHover,
  onCellSelect,
  selectedCell,
  panOnSpace,
  drawExtra,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const dpr = window.devicePixelRatio;

  const spacePressed = useSpacePressed();
  const isMouseDown = useIsMouseDown();

  const enablePan = !panOnSpace || spacePressed;

  const { setContainer, panZoomHandlers, setZoom, setPan, pan, zoom } =
    usePanZoom({
      minZoom: 1,
      maxZoom: 10,
      zoomSensitivity: 0.007,
      enablePan,
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

    const scale = Math.min(widthBasedSize, heightBasedSize);
    setScale(scale);

    // copy-paste of recenter code (it's hard to add a dependency because of hook rules)
    setPan({
      x: (canvas.width - grid.width * scale) / (2 * dpr),
      y: (canvas.height - grid.height * scale) / (2 * dpr),
    });
    setZoom(1);
  }, [grid.width, grid.height]);

  useEffect(() => {
    initCanvas();
  }, [initCanvas]);

  const recenter = useCallback(() => {
    if (!scale || !canvasRef.current) return;
    const canvas = canvasRef.current;
    setZoom(1);
    setPan({
      x: (canvas.width - grid.width * scale) / (2 * dpr),
      y: (canvas.height - grid.height * scale) / (2 * dpr),
    });
  }, [scale, setPan, setZoom]);

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

    try {
      drawGrid({
        grid,
        context,
        drawer,
      });

      drawExtra?.(context);
    } catch (e) {
      context.resetTransform();
      context.fillStyle = "black";
      context.globalAlpha = 1;
      context.fillRect(0, 0, canvas.width, canvas.height);
      context.font = "60px Arial";
      context.fillStyle = "red";
      context.fillText("Error drawing grid. Check console.", 80, 80);
      console.error(e);
      return;
    }

    context.lineWidth = 0.03;

    if (hoveredCell && grid.cellInGrid(hoveredCell)) {
      context.strokeStyle = "white";
      const margin = 0.03;
      context.roundRect(
        hoveredCell.c + margin,
        hoveredCell.r + margin,
        1 - 2 * margin,
        1 - 2 * margin,
        0.1
      );
      context.stroke();
    }

    if (selectedCell && grid.cellInGrid(selectedCell)) {
      context.strokeStyle = "blue";
      context.roundRect(selectedCell.c, selectedCell.r, 1, 1, 0.1);
      context.stroke();
    }
  }, [drawer, grid, transform, hoveredCell, selectedCell, drawExtra]);

  useCallOnWindowResize(initCanvas);
  useCallOnElementResize(canvasRef.current, initCanvas);

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
      onCellSelect(cell);
    },
    [zoom, grid, onCellSelect, cellFromMouseEvent]
  );

  return (
    <div
      className="relative flex h-full w-full overflow-hidden"
      {...panZoomHandlers}
      ref={setContainer}
    >
      <canvas
        ref={canvasRef}
        onDoubleClick={recenter}
        // Canvas takes all available space.
        className={clsx(
          "h-full w-full overflow-hidden",
          enablePan && (isMouseDown ? "cursor-grabbing" : "cursor-grab")
        )}
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

export const MapViewer: FC<Props> = (props) => {
  if (typeof window === "undefined") {
    return (
      <div className="relative flex h-full w-full overflow-hidden">
        <canvas className="h-full w-full overflow-hidden" />
      </div>
    );
  }

  return <MapViewerBrowserOnly {...props} />;
};
