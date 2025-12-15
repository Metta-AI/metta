"use client";
import clsx from "clsx";
import {
  FC,
  use,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { usePanZoom } from "@/hooks/use-pan-and-zoom";
import { useIsMouseDown } from "@/hooks/useIsMouseDown";
import { Cell } from "@/lib/MettaGrid";

import { DebugInfo } from "./DebugInfo";
import {
  useCallOnElementResize,
  useCallOnWindowResize,
  useDrawer,
  useSpacePressed,
} from "./hooks";
import { MapViewerProps } from "./index";
import { MapViewerContext } from "./MapViewerContext";
import { MapViewerCornerMenu } from "./MapViewerCornerMenu";

export const MapViewerBrowserOnly: FC<MapViewerProps> = ({
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
  const maxZoom = Math.max(grid.width, grid.height) * 1.5;

  const { setContainer, panZoomHandlers, setZoom, setPan, pan, zoom } =
    usePanZoom({
      minZoom: 1,
      maxZoom,
      zoomSensitivity: 0.005,
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
  }, [dpr, grid.width, grid.height, setPan, setZoom]);

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
  }, [dpr, grid.width, grid.height, scale, setPan, setZoom]);

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

  const drawGrid = useCallback(() => {
    if (!drawer || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");
    if (!context) return;

    context.resetTransform();
    context.fillStyle = "#eee";
    context.fillRect(0, 0, canvas.width, canvas.height);

    context.setTransform(transform);

    try {
      drawer.drawGrid(context, grid);
      drawExtra?.(context);

      // Draw hover overlay on top of the grid
      context.save();
      context.setTransform(transform);
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
      context.restore();
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
  }, [drawer, grid, transform, drawExtra, hoveredCell, selectedCell]);

  useCallOnWindowResize(initCanvas);
  useCallOnElementResize(canvasRef.current, initCanvas);

  useEffect(() => {
    drawGrid();
  }, [drawGrid]);

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
    [transform, grid, dpr]
  );

  const onMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const cell = cellFromMouseEvent(e);

      // Do not redraw if hovering the same cell
      if (hoveredCell?.r === cell?.r && hoveredCell?.c === cell?.c) {
        return;
      }

      setHoveredCell(cell);
      onCellHover?.(cell);
    },
    [hoveredCell, cellFromMouseEvent, onCellHover]
  );

  const onMouseClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!onCellSelect) return;

      const cell = cellFromMouseEvent(e);
      onCellSelect(cell);
    },
    [onCellSelect, cellFromMouseEvent]
  );

  const { showDebugInfo, showHoverInfo } = use(MapViewerContext);

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
      <div
        className="absolute top-2 right-2"
        onClick={(e) => e.stopPropagation()}
        onMouseMove={(e) => e.stopPropagation()}
      >
        <MapViewerCornerMenu />
      </div>
      {showDebugInfo && canvasRef.current && (
        <DebugInfo canvas={canvasRef.current} pan={pan} zoom={zoom} />
      )}
      {showHoverInfo && hoveredCell && (
        <div className="absolute right-0 bottom-0 z-10 bg-white/80 p-1 text-xs">
          <div>
            Hovered: {hoveredCell.r},{hoveredCell.c}
          </div>
          <div>Object: {grid.object(hoveredCell)?.name ?? "empty"}</div>
        </div>
      )}
    </div>
  );
};
