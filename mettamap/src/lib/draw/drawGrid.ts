import { MettaGrid } from "@/lib/MettaGrid";

import { Drawer } from "../../lib/draw/Drawer";

export async function drawGrid({
  grid,
  canvas,
  drawer,
  cellSize,
  // selectedCell,
}: {
  grid: MettaGrid;
  canvas: HTMLCanvasElement;
  drawer: Drawer;
  cellSize: number;
}) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  // Clear canvas
  ctx.fillStyle = "rgb(6, 24, 24)";
  ctx.fillRect(0, 0, cellSize * grid.width, cellSize * grid.height);

  // Draw grid lines
  ctx.strokeStyle = "#aaa";
  ctx.lineWidth = 0.5;

  // Draw vertical grid lines
  for (let x = 0; x <= grid.width; x++) {
    ctx.beginPath();
    ctx.moveTo(x * cellSize, 0);
    ctx.lineTo(x * cellSize, canvas.height);
    ctx.stroke();
  }

  // Draw horizontal grid lines
  for (let y = 0; y <= grid.height; y++) {
    ctx.beginPath();
    ctx.moveTo(0, y * cellSize);
    ctx.lineTo(canvas.width, y * cellSize);
    ctx.stroke();
  }

  // Draw the map
  for (const object of grid.objects) {
    drawer.drawObject(
      object,
      ctx,
      object.c * cellSize,
      object.r * cellSize,
      cellSize
    );
  }
}
