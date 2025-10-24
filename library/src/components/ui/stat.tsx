import * as React from "react";

import { cn } from "@/lib/utils";

interface StatProps extends React.HTMLAttributes<HTMLDivElement> {
  label: string;
  value: React.ReactNode;
  helperText?: string;
  trend?: "up" | "down" | null;
  trendText?: string;
}

export const Stat = React.forwardRef<HTMLDivElement, StatProps>(
  (
    { label, value, helperText, trend = null, trendText, className, ...props },
    ref
  ) => {
    return (
      <div
        ref={ref}
        className={cn(
          "border-border bg-card text-card-foreground rounded-lg border px-4 py-3 shadow-sm",
          className
        )}
        {...props}
      >
        <p className="text-muted-foreground text-xs font-medium tracking-wide uppercase">
          {label}
        </p>
        <div className="mt-1 flex items-baseline justify-between gap-2">
          <div className="text-2xl font-semibold">{value}</div>
          {trend && (
            <span
              className={cn(
                "flex items-center gap-1 text-xs font-medium",
                trend === "up" ? "text-emerald-600" : "text-rose-600"
              )}
            >
              <span aria-hidden> {trend === "up" ? "▲" : "▼"}</span>
              {trendText}
            </span>
          )}
        </div>
        {helperText && (
          <p className="text-muted-foreground mt-1 text-xs">{helperText}</p>
        )}
      </div>
    );
  }
);

Stat.displayName = "Stat";
