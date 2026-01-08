"use client";
import { FC, PropsWithChildren } from "react";
import { ErrorBoundary as ReactErrorBoundary } from "react-error-boundary";

import { ErrorFallback } from "./ErrorFallback";

export const ErrorBoundary: FC<
  PropsWithChildren<{ resetKeys?: unknown[] }>
> = ({ children, resetKeys }) => {
  return (
    <ReactErrorBoundary FallbackComponent={ErrorFallback} resetKeys={resetKeys}>
      {children}
    </ReactErrorBoundary>
  );
};
