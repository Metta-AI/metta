"use client";
import { createContext } from "react";

export const YamlContext = createContext<{
  isSelected?: (key: string, value: string) => boolean;
  onSelectLine?: (key: string, value: string) => void;
  unsetFields: Set<string>;
  kind?: string;
}>({ unsetFields: new Set() });
