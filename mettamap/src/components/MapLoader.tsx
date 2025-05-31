'use client';

import { FC, useEffect } from "react";
import { ExtendedMapViewer } from "./MapFileViewer";
import { getStoredMap } from "@/server/api";
import { useState } from "react";
import { MapData } from "@/server/types";

export const MapLoader: FC<{ mapUrl: string }> = ({ mapUrl }) => {
  const [map, setMap] = useState<MapData | null>(null);
  useEffect(() => {
    getStoredMap(mapUrl).then((map) => {
      setMap(map);
    });
  }, [mapUrl]);

  if (!map) {
    return <div>Loading...</div>;
  }

  return <ExtendedMapViewer mapData={map} />;
}
