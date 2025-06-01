"use client";

import { FC, useEffect, useState } from "react";

import { getStoredMap } from "@/server/api";
import { MapData } from "@/server/types";

import { ExtendedMapViewer } from "./MapFileViewer";

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
};
