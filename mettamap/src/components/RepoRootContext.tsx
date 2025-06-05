"use client";
import { createContext, FC, PropsWithChildren } from "react";

export const RepoRootContext = createContext<string | null>(null);

export const RepoRootProvider: FC<PropsWithChildren<{ root: string }>> = ({
  children,
  root,
}) => {
  return (
    <RepoRootContext.Provider value={root}>{children}</RepoRootContext.Provider>
  );
};
