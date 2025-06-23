import { MettaGrid } from "@/lib/MettaGrid";

import { Drawer } from "../../lib/draw/Drawer";

const BACKGROUND_COLOR = "#cfa970";

export async function drawGrid({
  grid,
  context: ctx,
  drawer,
}: {
  grid: MettaGrid;
  context: CanvasRenderingContext2D;
  drawer: Drawer;
}) {
  // Clear drawing area
  ctx.fillStyle = BACKGROUND_COLOR;
  ctx.fillRect(0, 0, grid.width, grid.height);

  // Draw grid lines
  ctx.strokeStyle = "#aaa";
  ctx.lineWidth = 0.5;

  // Draw vertical grid lines
  for (let x = 0; x <= grid.width; x++) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, grid.height);
    ctx.stroke();
  }

  // Draw horizontal grid lines
  for (let y = 0; y <= grid.height; y++) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(grid.width, y);
    ctx.stroke();
  }

  // Draw the map
  for (const object of grid.objects) {
    drawer.drawObject(object, ctx, object.c, object.r, 1);
  }
}
