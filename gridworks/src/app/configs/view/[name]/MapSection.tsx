"use client";
import { FC } from "react";
import { ErrorBoundary } from "react-error-boundary";

import { Button } from "@/components/Button";
import { ErrorFallback } from "@/components/ErrorFallback";
import { StorableMapViewer } from "@/components/StorableMapViewer";
import { useMapFromConfig } from "@/hooks/useMapFromConfig";
import { Config } from "@/lib/api";

export const MapSection: FC<{ cfg: Config; name?: string }> = ({
  cfg,
  name,
}) => {
  const { map, reload } = useMapFromConfig(cfg, name);

  return (
    <section className="mb-8">
      <div className="mb-4 flex items-center gap-1">
        <h2 className="text-xl font-bold">Generated Map</h2>
        <Button onClick={reload} size="sm">
          Regenerate
        </Button>
      </div>
      {map.type === "loading" && (
        <div className="h-screen w-full bg-gray-100" />
      )}
      {map.type === "map" && (
        <ErrorBoundary FallbackComponent={ErrorFallback}>
          <StorableMapViewer map={map.map} />
        </ErrorBoundary>
      )}
      {map.type === "error" && (
        <pre className="text-sm text-red-500">
          Error loading map: {map.error.message}
        </pre>
      )}
    </section>
  );
};
