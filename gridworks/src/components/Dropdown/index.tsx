// This file and other Dropdown components are ported from: https://www.npmjs.com/package/@quri/ui
//
// MIT License
//
// Copyright (c) 2020 Foretold
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

"use client";
import {
  arrow,
  flip,
  FloatingPortal,
  offset,
  useClick,
  useDismiss,
  useFloating,
  useInteractions,
} from "@floating-ui/react";
import {
  FC,
  PropsWithChildren,
  ReactNode,
  useCallback,
  useRef,
  useState,
} from "react";

import { DropdownContext } from "./DropdownContext";

type Props = PropsWithChildren<{
  // `close` option is for backward compatibility; new code should use `useCloseDropdown` instead.
  render(options: { close(): void }): ReactNode;
}>;

export const Dropdown: FC<Props> = ({ render, children }) => {
  const [isOpen, setIsOpen] = useState(false);

  const arrowRef = useRef<HTMLDivElement | null>(null);

  const { x, y, strategy, refs, context } = useFloating({
    open: isOpen,
    onOpenChange: setIsOpen,
    placement: "bottom-start",
    middleware: [offset(4), flip(), arrow({ element: arrowRef })],
  });

  const click = useClick(context);
  const dismiss = useDismiss(context);

  const { getReferenceProps, getFloatingProps } = useInteractions([
    click,
    dismiss,
  ]);

  const closeDropdown = useCallback(() => {
    setIsOpen(false);
  }, []);

  const renderTooltip = () => (
    <FloatingPortal>
      <div
        ref={refs.setFloating}
        className="z-30 overflow-hidden rounded-md border border-slate-300 bg-white shadow-xl"
        style={{
          position: strategy,
          top: y ?? 0,
          left: x ?? 0,
        }}
        {...getFloatingProps()}
      >
        {render({ close: closeDropdown })}
      </div>
    </FloatingPortal>
  );

  return (
    <DropdownContext value={{ closeDropdown }}>
      <div ref={refs.setReference} {...getReferenceProps()}>
        {children}
      </div>
      {isOpen ? renderTooltip() : null}
    </DropdownContext>
  );
};
