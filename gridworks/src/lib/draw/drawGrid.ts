import { MettaGrid } from "@/lib/MettaGrid";

import { Drawer } from "../../lib/draw/Drawer";

const BACKGROUND_COLOR = "#cfa970";

export function drawGrid({
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

  // Draw the map
  for (const object of grid.objects) {
    if (object.name === "wall") {
      // in mettascope, walls have many different types depending on the surrounding terrain.
      // until we support that, we just draw a single wall type, and highlight it with color.
      ctx.fillStyle = "#c0bcb8";
      ctx.fillRect(object.c, object.r, 1, 1);
    }
    drawer.drawObject(object.name, ctx, object.c, object.r, 1);
  }

  // Draw grid lines
  {
    ctx.strokeStyle = "black";
    ctx.globalAlpha = 0.2;
    ctx.lineCap = "square";
    ctx.lineWidth = 0.02;

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
    ctx.globalAlpha = 1;
  }
}
