"use client";
import dynamic from "next/dynamic";
import { FC } from "react";

import { Cell, MettaGrid } from "@/lib/MettaGrid";

import { MapViewerContextProvider } from "./MapViewerContext";

type CellHandler = (cell: Cell | undefined) => void;

// Useful for external components to draw on top of the map, e.g. for drawing areas from the scene tree.
export type MapDrawer = (context: CanvasRenderingContext2D) => void;

export type MapViewerProps = {
  grid: MettaGrid;
  onCellHover?: CellHandler;
  selectedCell?: Cell;
  onCellSelect?: CellHandler;
  panOnSpace?: boolean;
  drawExtra?: MapDrawer;
};

const MapViewerBrowserOnly = dynamic(
  () =>
    import("./MapViewerBrowserOnly").then((mod) => mod.MapViewerBrowserOnly),
  {
    ssr: false,
    loading: () => (
      <div className="relative flex h-full w-full overflow-hidden">
        <canvas className="h-full w-full overflow-hidden" />
      </div>
    ),
  }
);

export const MapViewer: FC<MapViewerProps> = (props) => {
  return (
    <MapViewerContextProvider>
      <MapViewerBrowserOnly {...props} />
    </MapViewerContextProvider>
  );
};
