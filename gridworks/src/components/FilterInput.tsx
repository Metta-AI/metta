"use client";
import clsx from "clsx";
import { type FC, useEffect, useRef } from "react";

export const FilterInput: FC<{
  className?: string;
  focus?: boolean;
  placeholder?: string;
  value: string;
  onBlur?: () => void;
  onChange: (value: string) => void;
}> = ({
  className,
  focus = false,
  placeholder = "Filter...",
  value,
  onBlur,
  onChange,
}) => {
  const inputRef = useRef<HTMLInputElement>(null);
  const prevFocusRef = useRef(false);

  useEffect(() => {
    if (focus && !prevFocusRef.current) {
      // Render on next tick
      const id = setTimeout(() => {
        console.dir(inputRef.current);
        inputRef.current?.focus();
        inputRef.current?.select();
      }, 0);

      return () => clearTimeout(id);
    }
    prevFocusRef.current = focus;
  }, [focus]);

  return (
    <input
      ref={inputRef}
      type="text"
      value={value}
      placeholder={placeholder}
      className={clsx("w-full rounded border border-gray-300 p-2", className)}
      onChange={(e) => onChange(e.target.value)}
      onBlur={onBlur}
    />
  );
};
