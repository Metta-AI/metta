/* eslint-disable @typescript-eslint/no-explicit-any */
/**
 * Code from https://github.com/wouterraateland/use-pan-and-zoom.
 *
 * MIT License
 *
 * Copyright (c) 2019 Wouter Raateland
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

// Forked because the original code had bugs. In particular, its useForceUpdate function used a boolean instead of a number,
// which is not compatible with the way modern React batches state updates.
import { useCallback, useMemo, useRef } from "react";

import { useForceUpdate } from "./use-force-update";
import { useGetSet } from "./use-get-set";
import {
  clamp,
  dist,
  getPositionOnElement,
  noop,
  position,
  transform,
} from "./utils";

export function usePanZoom({
  enablePan = true,
  enableZoom = true,
  requireCtrlToZoom = false,
  disableWheel = false,
  panOnDrag = true,
  preventClickOnPan = true,
  zoomSensitivity = 0.01,
  minZoom = 0,
  maxZoom = Infinity,
  minX = -Infinity,
  maxX = Infinity,
  minY = -Infinity,
  maxY = Infinity,
  initialZoom = 1,
  initialPan = { x: 0, y: 0 },
  onPanStart = noop,
  onPan = noop,
  onPanEnd = noop,
  onZoom = noop,
}: {
  enablePan?: boolean;
  enableZoom?: boolean;
  requireCtrlToZoom?: boolean;
  disableWheel?: boolean;
  panOnDrag?: boolean;
  preventClickOnPan?: boolean;
  zoomSensitivity?: number;
  minZoom?: number;
  maxZoom?: number;
  minX?: number;
  maxX?: number;
  minY?: number;
  maxY?: number;
  initialZoom?: number;
  initialPan?: position;
  onPanStart?: (touches: position[], transform: transform) => void;
  onPan?: (touches: position[], transform: transform) => void;
  onPanEnd?: () => void;
  onZoom?: (transform: transform) => void;
} = {}) {
  const containerRef = useRef<HTMLElement | null>(null);
  const forceUpdate = useForceUpdate();
  const [getState, setState] = useGetSet<{
    wasPanning: boolean;
    isPanning: boolean;
    prevPointers: position[];
    prevZoom: number;
    center: position;
    transform: transform;
  }>({
    wasPanning: false,
    isPanning: false,
    prevPointers: [],
    prevZoom: 1,
    center: { x: 0, y: 0 },
    transform: { ...initialPan, zoom: initialZoom },
  });

  const clampX = useMemo(() => clamp(minX, maxX), [minX, maxX]);
  const clampY = useMemo(() => clamp(minY, maxY), [minY, maxY]);
  const clampZoom = useMemo(() => clamp(minZoom, maxZoom), [minZoom, maxZoom]);

  const setTransform = useCallback(
    (v: transform | ((current: transform) => transform)) => {
      const container = containerRef.current;
      const state = getState();
      if (container) {
        state.transform = typeof v === "function" ? v(state.transform) : v;
        state.center.x =
          (container.offsetWidth / 2 - state.transform.x) /
          state.transform.zoom;
        state.center.y =
          (container.offsetHeight / 2 - state.transform.y) /
          state.transform.zoom;

        setState(state);
        forceUpdate();
      }
      return state.transform;
    },
    [getState, setState, forceUpdate]
  );

  const setPan = useCallback(
    (value: ((current: position) => position) | position) =>
      setTransform(({ x, y, zoom }) => {
        const newPan = typeof value === "function" ? value({ x, y }) : value;

        return {
          x: clampX(newPan.x),
          y: clampY(newPan.y),
          zoom,
        };
      }),
    [clampX, clampY, setTransform]
  );

  const setZoom = useCallback(
    (value: number | ((current: number) => number), maybeCenter?: position) => {
      const container = containerRef.current;
      if (container) {
        setTransform(({ x, y, zoom }) => {
          const newZoom = clampZoom(
            typeof value === "function" ? value(zoom) : value
          );

          const center = maybeCenter
            ? {
                x: maybeCenter.x - container.offsetWidth / 2,
                y: maybeCenter.y - container.offsetHeight / 2,
              }
            : { x: 0, y: 0 };

          return {
            x: clampX(x + ((center.x - x) * (zoom - newZoom)) / zoom),
            y: clampY(y + ((center.y - y) * (zoom - newZoom)) / zoom),
            zoom: newZoom,
          };
        });
      }
    },
    [clampX, clampY, clampZoom, setTransform]
  );

  const startPanZoom = useCallback(
    (pointers: position[]) => {
      if (enablePan) {
        const state = getState();
        state.prevPointers = pointers;
        state.isPanning = true;
        setState(state);
        onPanStart(pointers, state.transform);
      }
    },
    [enablePan, onPanStart, setState, getState]
  );

  const movePanZoom = useCallback(
    (pointers: position[]) => {
      const state = getState();
      if (state.isPanning) {
        state.wasPanning = true;

        const prevPointers = state.prevPointers;
        state.prevPointers = pointers;
        setState(state);

        let dx = 0,
          dy = 0;
        const l = Math.min(pointers.length, prevPointers.length, 2);

        for (let i = 0; i < l; i++) {
          dx += pointers[i].x - prevPointers[i].x;
          dy += pointers[i].y - prevPointers[i].y;
        }
        dx /= l;
        dy /= l;
        const scale =
          l === 2
            ? dist(pointers[0], pointers[1]) /
              dist(prevPointers[0], prevPointers[1])
            : 1;

        setPan(({ x, y }) => ({
          x: x + dx,
          y: y + dy,
        }));
        setZoom((zoom) => zoom * scale);

        forceUpdate();
        onPan(pointers, getState().transform);
      }
    },
    [onPan, setPan, setZoom, getState, setState, forceUpdate]
  );

  const endPanZoom = useCallback(() => {
    const state = getState();
    if (state.isPanning) {
      setState((state) => ({ ...state, isPanning: false }));
      onPanEnd();
    }
  }, [onPanEnd, getState, setState]);

  const onClickCapture = useCallback(
    (event: React.MouseEvent) => {
      const state = getState();
      if (preventClickOnPan && state.wasPanning) {
        setState((state) => ({ ...state, wasPanning: false }));
        event.stopPropagation();
      }
    },
    [preventClickOnPan, getState, setState]
  );

  const onWheel = useCallback(
    (event: WheelEvent) => {
      if (enableZoom && containerRef.current) {
        event.preventDefault();
        if (!requireCtrlToZoom || event.ctrlKey) {
          const pointerPosition = getPositionOnElement(containerRef.current, {
            x: event.pageX,
            y: event.pageY,
          });

          let deltaY = event.deltaY;
          if (event.deltaMode === 1) {
            deltaY *= 15;
          }

          setZoom(
            (zoom) => zoom * Math.pow(1 - zoomSensitivity, deltaY),
            pointerPosition
          );

          onZoom(getState().transform);
        } else {
          setPan(({ x, y }) => ({
            x: x - event.deltaX,
            y: y - event.deltaY,
          }));
        }
      }
    },
    [
      enableZoom,
      requireCtrlToZoom,
      zoomSensitivity,
      onZoom,
      setPan,
      setZoom,
      getState,
    ]
  );

  const onGestureStart = useCallback(
    (event: any) => {
      event.preventDefault();
      setState((state) => ({ ...state, prevZoom: state.transform.zoom }));
    },
    [setState]
  );

  const onGesture = useCallback(
    (event: any) => {
      event.preventDefault();

      const pointerPosition = getPositionOnElement(containerRef.current, {
        x: event.pageX,
        y: event.pageY,
      });
      setZoom(getState().prevZoom * event.scale, pointerPosition);
      onZoom(getState().transform);
    },
    [setZoom, onZoom, getState]
  );

  const setContainer = useCallback(
    (el: HTMLElement | null) => {
      if (el) {
        if (!disableWheel) {
          el.addEventListener("wheel", onWheel);
        }
        el.addEventListener("gesturestart", onGestureStart);
        el.addEventListener("gesturechange", onGesture);
        el.addEventListener("gestureend", onGesture);
      } else if (containerRef.current) {
        return () => {
          const container = containerRef.current;
          if (container) {
            if (!disableWheel) {
              container.removeEventListener("wheel", onWheel);
            }
            container.removeEventListener("gesturestart", onGestureStart);
            container.removeEventListener("gesturechange", onGesture);
            container.removeEventListener("gestureend", onGesture);
          }
        };
      }
      containerRef.current = el;
    },
    [onWheel, onGestureStart, onGesture, disableWheel]
  );

  const onTouchStart = useCallback(
    (event: React.TouchEvent) =>
      startPanZoom(
        Array.from(event.touches).map(({ pageX, pageY }) => ({
          x: pageX,
          y: pageY,
        }))
      ),
    [startPanZoom]
  );
  const onTouchMove = useCallback(
    (event: React.TouchEvent) =>
      movePanZoom(
        Array.from(event.touches).map(({ pageX, pageY }) => ({
          x: pageX,
          y: pageY,
        }))
      ),
    [movePanZoom]
  );
  const onTouchEnd = useCallback(
    (event: React.TouchEvent) => {
      if (event.touches.length === 0) {
        endPanZoom();
      } else {
        startPanZoom(
          Array.from(event.touches).map(({ pageX, pageY }) => ({
            x: pageX,
            y: pageY,
          }))
        );
      }
    },
    [endPanZoom, startPanZoom]
  );
  const onMouseDown = useCallback(
    (event: React.MouseEvent) =>
      startPanZoom([{ x: event.pageX, y: event.pageY }]),
    [startPanZoom]
  );
  const onMouseMove = useCallback(
    (event: React.MouseEvent) =>
      movePanZoom([{ x: event.pageX, y: event.pageY }]),
    [movePanZoom]
  );

  const state = getState();
  return {
    container: containerRef.current,
    setContainer,
    transform: `translate3D(${state.transform.x}px, ${state.transform.y}px, 0) scale(${state.transform.zoom})`,
    center: state.center,
    pan: { x: state.transform.x, y: state.transform.y },
    zoom: state.transform.zoom,
    setPan,
    setZoom,
    panZoomHandlers: panOnDrag
      ? {
          onTouchStart,
          onTouchMove,
          onTouchEnd: onTouchEnd,
          onTouchCancel: onTouchEnd,
          onMouseDown,
          onMouseMove,
          onMouseUp: endPanZoom,
          onMouseLeave: endPanZoom,
          onClickCapture,
        }
      : {},
  };
}
