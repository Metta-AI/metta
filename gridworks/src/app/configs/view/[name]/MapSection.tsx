"use client";
import { FC } from "react";

import { ReloadableMapViewer } from "@/components/ReloadableMapViewer";
import { useLoadMapFromConfig } from "@/hooks/useLoadMap";
import { MakerConfig } from "@/lib/api";

export const MapSection: FC<{ cfg: MakerConfig; name?: string }> = ({
  cfg,
  name,
}) => {
  const { mapState, reload } = useLoadMapFromConfig(cfg, name);

  return <ReloadableMapViewer mapState={mapState} reload={reload} />;
};
