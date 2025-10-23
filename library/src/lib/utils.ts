import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export * from "./utils/date";
export * from "./utils/text";
export * from "./utils/validation";
export * from "./utils/errors";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
