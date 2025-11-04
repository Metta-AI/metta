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
  const { mapState, reload } = useMapFromConfig(cfg, name);

  return (
    <section className="mb-8">
      <div className="mb-4 flex items-center gap-2">
        <h2 className="text-xl font-bold">Generated Map</h2>
        <Button onClick={reload} size="sm" disabled={mapState.loading}>
          Regenerate
        </Button>
      </div>
      <div className="relative">
        {mapState.type === "init" && (
          <div className="h-screen w-full bg-gray-100" />
        )}
        {mapState.type === "map" && (
          <ErrorBoundary FallbackComponent={ErrorFallback}>
            <StorableMapViewer map={mapState.map} />
          </ErrorBoundary>
        )}
        {mapState.loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-white/50">
            <div className="h-10 w-10 animate-spin rounded-full border-t-2 border-b-2 border-gray-900" />
          </div>
        )}
        {mapState.type === "error" && (
          <pre className="text-sm text-red-500">
            Error loading map: {mapState.error.message}
          </pre>
        )}
      </div>
    </section>
  );
};
