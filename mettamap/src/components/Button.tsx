"use client";
import { FC } from "react";

import clsx from "clsx";

export const Button: FC<{
  onClick?: () => void;
  children: React.ReactNode;
  theme?: "primary" | "secondary";
  type?: "button" | "submit";
}> = ({ onClick, children, theme = "secondary", type = "button" }) => {
  return (
    <button
      className={clsx(
        "cursor-pointer rounded-md border-2 px-4 py-1 text-sm",
        theme === "primary" &&
          "border-blue-500 bg-blue-500 text-white hover:bg-blue-600",
        theme === "secondary" &&
          "border-blue-400 text-blue-500 hover:bg-blue-100"
      )}
      onClick={onClick}
      type={type}
    >
      {children}
    </button>
  );
};
