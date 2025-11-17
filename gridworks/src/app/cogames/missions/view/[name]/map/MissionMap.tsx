"use client";
import { FC } from "react";

import { ReloadableMapViewer } from "@/components/ReloadableMapViewer";
import { useLoadMapFromMission } from "@/hooks/useLoadMap";

import { useVariants } from "../useVariants";

export const MissionMap: FC<{ name: string }> = ({ name }) => {
  const [variants] = useVariants();

  const { mapState, reload } = useLoadMapFromMission(name, variants);

  return (
    <div className="pt-4">
      <ReloadableMapViewer mapState={mapState} reload={reload} />
    </div>
  );
};
