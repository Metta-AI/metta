"use client";

import { FC, useEffect, useState } from "react";

import { getStoredMap, StorableMap } from "@/lib/api";

import { StorableMapViewer } from "./StorableMapViewer";

export const MapLoader: FC<{ mapUrl: string; filterable?: boolean }> = ({
  mapUrl,
  filterable = true,
}) => {
  const [map, setMap] = useState<StorableMap | null>(null);
  useEffect(() => {
    getStoredMap(mapUrl).then((map) => {
      setMap(map);
    });
  }, [mapUrl]);

  if (!map) {
    return <div>Loading...</div>;
  }

  return <StorableMapViewer url={mapUrl} map={map} filterable={filterable} />;
};
