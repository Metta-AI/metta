"use client";
import { FC, useEffect, useState } from "react";

import { Button } from "@/components/Button";
import { StorableMapViewer } from "@/components/StorableMapViewer";
import {
  getMettagridCfgMap,
  MaybeStorableMap,
  MettagridCfgFile,
} from "@/lib/api";

export const MapSection: FC<{ cfg: MettagridCfgFile }> = ({ cfg }) => {
  const [id, setId] = useState(0);

  const [maybeMap, setMaybeMap] = useState<MaybeStorableMap | null>(null);
  useEffect(() => {
    getMettagridCfgMap(cfg.metadata.path).then(setMaybeMap);
  }, [cfg.metadata.path, id]);

  if (maybeMap === null) {
    return <div className="h-screen w-full bg-gray-100" />;
  }

  return (
    <section className="mb-8">
      <div className="mb-4 flex items-center gap-1">
        <h2 className="text-xl font-bold">Generated Map</h2>
        <Button onClick={() => setId(id + 1)} size="sm">
          Regenerate
        </Button>
      </div>
      {maybeMap.type === "map" && <StorableMapViewer map={maybeMap.data} />}
      {maybeMap.type === "error" && (
        <pre className="text-sm text-red-500">
          Error loading map: {maybeMap.error}
        </pre>
      )}
    </section>
  );
};
