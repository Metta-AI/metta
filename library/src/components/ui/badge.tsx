import React from "react";
import { clsx } from "clsx";

export interface BadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: "default" | "secondary" | "destructive" | "outline";
}

const Badge = React.forwardRef<HTMLDivElement, BadgeProps>(
  ({ className, variant = "default", ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={clsx(
          "focus:ring-ring inline-flex items-center rounded-md px-2.5 py-0.5 text-xs font-medium transition-colors focus:ring-2 focus:ring-offset-2 focus:outline-none",
          {
            "border-transparent bg-neutral-100 text-neutral-900 hover:bg-neutral-200":
              variant === "default",
            "border-transparent bg-neutral-200 text-neutral-900 hover:bg-neutral-300":
              variant === "secondary",
            "border-transparent bg-red-100 text-red-900 hover:bg-red-200":
              variant === "destructive",
            "border border-neutral-300 bg-transparent text-neutral-900 hover:bg-neutral-100":
              variant === "outline",
          },
          className
        )}
        {...props}
      />
    );
  }
);
Badge.displayName = "Badge";

export { Badge };
