"use client";
import { FC } from "react";

import { Button } from "@/components/Button";
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
      {map.type === "map" && <StorableMapViewer map={map.map} />}
      {map.type === "error" && (
        <pre className="text-sm text-red-500">
          Error loading map: {map.error.message}
        </pre>
      )}
    </section>
  );
};
