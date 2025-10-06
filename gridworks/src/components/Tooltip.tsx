"use client";
import {
  flip,
  useFloating,
  useHover,
  useInteractions,
} from "@floating-ui/react";
import { FC, useState } from "react";

export const Tooltip: FC<{
  children: React.ReactNode;
  render: () => React.ReactNode;
}> = ({ children, render }) => {
  const [isOpen, setIsOpen] = useState(false);

  const { refs, floatingStyles, context } = useFloating({
    open: isOpen,
    onOpenChange: setIsOpen,
    placement: "top-start",
    middleware: [flip()],
  });

  const hover = useHover(context);

  const { getReferenceProps, getFloatingProps } = useInteractions([hover]);
  return (
    <>
      <div ref={refs.setReference} {...getReferenceProps()}>
        {children}
      </div>
      {isOpen && (
        <div
          ref={refs.setFloating}
          {...getFloatingProps({
            style: floatingStyles,
            className:
              "z-50 rounded-lg border border-zinc-400 bg-white px-3 py-1.5 text-sm shadow-xl",
          })}
        >
          {render()}
        </div>
      )}
    </>
  );
};
