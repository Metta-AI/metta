"use client";
import { FC } from "react";

import { useCopyTooltip } from "@/hooks/useCopyTooltip";

import { Button } from "./Button";

export const CopyTextButton: FC<{
  text: string | (() => string);
  children: React.ReactNode;
  theme?: "primary" | "secondary";
  type?: "button" | "submit";
  size?: "sm" | "md";
  disabled?: boolean;
}> = ({
  text,
  children,
  theme = "secondary",
  type = "button",
  size = "md",
  disabled = false,
}) => {
  const { onClick, floating, render } = useCopyTooltip(text);

  return (
    <div>
      <div ref={floating.refs.setReference}>
        <Button
          theme={theme}
          type={type}
          size={size}
          disabled={disabled}
          onClick={onClick}
        >
          {children}
        </Button>
      </div>
      {render()}
    </div>
  );
};
