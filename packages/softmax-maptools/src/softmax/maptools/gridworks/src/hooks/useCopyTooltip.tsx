import {
  arrow,
  autoUpdate,
  flip,
  FloatingArrow,
  FloatingPortal,
  offset,
  shift,
  useFloating,
} from "@floating-ui/react";
import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";

export function useCopyTooltip(text: string) {
  const [open, setOpen] = useState(false);
  const hideTimerRef = useRef<NodeJS.Timeout | null>(null);
  const arrowRef = useRef(null);
  const placement = "top";
  const durationMs = 700;

  // Refer to https://floating-ui.com/docs/react documentation for more details.
  const floating = useFloating({
    open,
    onOpenChange: setOpen,
    placement,
    whileElementsMounted: autoUpdate,
    middleware: [
      offset(8),
      flip(),
      shift({ padding: 6 }),
      arrow({ element: arrowRef }),
    ],
  });

  // Clean up any pending timers on unmount
  useEffect(
    () => () => {
      if (hideTimerRef.current) {
        clearTimeout(hideTimerRef.current);
      }
    },
    []
  );

  function showFor(ms: number) {
    setOpen(true);
    if (hideTimerRef.current) {
      clearTimeout(hideTimerRef.current);
    }
    hideTimerRef.current = setTimeout(() => setOpen(false), ms);
  }

  const onClick = async (e: React.MouseEvent<HTMLDivElement>) => {
    e.preventDefault();
    await navigator.clipboard.writeText(text);
    showFor(durationMs);
  };

  const render = () => (
    <FloatingPortal>
      <AnimatePresence>
        {open && (
          <motion.div
            ref={floating.refs.setFloating}
            role="tooltip"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.12 }}
            className="z-[1000]"
            style={{
              position: floating.strategy,
              top: floating.y ?? 0,
              left: floating.x ?? 0,
            }}
          >
            <div className="relative rounded-lg border border-zinc-700 bg-black px-3 py-1.5 text-sm text-white shadow-xl">
              Copied to clipboard
              <FloatingArrow ref={arrowRef} context={floating.context} />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </FloatingPortal>
  );

  return {
    onClick,
    floating,
    render,
  };
}
