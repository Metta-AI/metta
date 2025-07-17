import { FC } from "react";

export const DebugInfo: FC<{
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
