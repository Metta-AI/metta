"use client";
import clsx from "clsx";
import { FC } from "react";

export const Button: FC<{
  onClick?: () => void;
  children: React.ReactNode;
  theme?: "primary" | "secondary";
  type?: "button" | "submit";
  size?: "sm" | "md";
  disabled?: boolean;
}> = ({
  onClick,
  children,
  theme = "secondary",
  type = "button",
  size = "md",
  disabled = false,
}) => {
  return (
    <button
      className={clsx(
        "cursor-pointer rounded-md border-2",
        size === "sm" && "px-2 py-0.5 text-xs",
        size === "md" && "px-4 py-1 text-sm",
        theme === "primary" &&
          "border-blue-500 bg-blue-500 text-white hover:bg-blue-600",
        theme === "secondary" &&
          "border-blue-400 text-blue-500 hover:bg-blue-100",
        disabled && "opacity-50"
      )}
      onClick={onClick}
      type={type}
      disabled={disabled}
    >
      {children}
    </button>
  );
};
