"use client";
import {
  createContext,
  FC,
  PropsWithChildren,
  useEffect,
  useState,
} from "react";

import { getJsonSchemas, JsonSchemas } from "@/lib/api/schemas";

export const JsonSchemasContext = createContext<{ schemas: JsonSchemas }>({
  schemas: { $defs: {} },
});

// Global context; loads schemas lazily.
export const JsonSchemasProvider: FC<PropsWithChildren> = ({ children }) => {
  const [schemas, setSchemas] = useState<JsonSchemas>({ $defs: {} });

  useEffect(() => {
    getJsonSchemas().then(setSchemas);
  }, []);
  return (
    <JsonSchemasContext value={{ schemas }}>{children}</JsonSchemasContext>
  );
};
