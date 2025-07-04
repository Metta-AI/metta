"use client";
import { FC, useEffect, useState } from "react";

import { Button } from "@/components/Button";
import { StorableMapViewer } from "@/components/StorableMapViewer";
import { getMettagridCfgMap, MettagridCfgFile, StorableMap } from "@/lib/api";

export const MapSection: FC<{ cfg: MettagridCfgFile }> = ({ cfg }) => {
  const [id, setId] = useState(0);

  type MapState =
    | {
        type: "loading";
      }
    | {
        type: "error";
        error: Error;
      }
    | {
        type: "map";
        map: StorableMap;
      };

  const [map, setMap] = useState<MapState>({ type: "loading" });
  useEffect(() => {
    setMap({ type: "loading" });
    getMettagridCfgMap(cfg.metadata.path)
      .then((map) => setMap({ type: "map", map }))
      .catch((e) => {
        console.error(e);
        setMap({ type: "error", error: e });
      });
  }, [cfg.metadata.path, id]);

  return (
    <section className="mb-8">
      <div className="mb-4 flex items-center gap-1">
        <h2 className="text-xl font-bold">Generated Map</h2>
        <Button onClick={() => setId(id + 1)} size="sm">
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
