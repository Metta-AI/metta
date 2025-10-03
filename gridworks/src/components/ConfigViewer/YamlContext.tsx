"use client";
import { createContext } from "react";

export const YamlContext = createContext<{
  isSelected?: (key: string, value: string) => boolean;
  onSelectLine?: (key: string, value: string) => void;
  unsetFields: Set<string>;
  kind?: string;
  topValue: unknown;
  showDebugInfo: boolean;
  setShowDebugInfo: (showDebugInfo: boolean) => void;
}>({
  unsetFields: new Set(),
  topValue: undefined,
  showDebugInfo: false,
  setShowDebugInfo: () => {},
});
