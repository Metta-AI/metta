import { useEffect, useState } from "react";

import { Drawer } from "@/lib/draw/Drawer";

export function useDrawer(): Drawer | undefined {
  const [drawer, setDrawer] = useState<Drawer | undefined>();
  useEffect(() => {
    Drawer.load().then(setDrawer);
  }, []);

  return drawer;
}

export function useCallOnWindowResize(callback: () => void) {
  useEffect(() => {
    window.addEventListener("resize", callback);
    return () => window.removeEventListener("resize", callback);
  }, [callback]);
}

export function useCallOnElementResize(
  element: HTMLElement | null,
  callback: () => void
) {
  useEffect(() => {
    if (!element) return;

    const observer = new ResizeObserver(() => {
      callback();
    });

    observer.observe(element);
    return () => observer.disconnect();
  }, [element, callback]);
}

export function useSpacePressed(): boolean {
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
