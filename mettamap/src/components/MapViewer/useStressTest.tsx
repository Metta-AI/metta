import { useEffect } from "react";

// Use this hook to stress test the canvas rendering performance.
// It will redraw the canvas 60 times per second when the canvas is visible on screen.
export function useStressTest(
  draw: () => void,
  canvas: HTMLCanvasElement | null
) {
  useEffect(() => {
    let interval: NodeJS.Timeout | null = null;

    // Only redraw if the canvas is visible on screen
    const observer = new IntersectionObserver((entries) => {
      const [entry] = entries;
      if (entry.isIntersecting) {
        console.log("visible");
        // Start animation loop when visible
        interval = setInterval(() => {
          console.log("drawing");
          draw();
        }, 100 / 60);
      } else {
        console.log("hidden");
        if (interval) {
          clearInterval(interval);
        }
        interval = null;
      }
    });

    if (canvas) {
      observer.observe(canvas);
    }

    return () => {
      if (canvas) {
        observer.disconnect();
      }
      if (interval) {
        clearInterval(interval);
      }
    };
  }, [draw, canvas]);
}
