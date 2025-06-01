import { FC } from "react";

import { ExtendedMapViewer } from "@/components/MapFileViewer";
import { getMettagridCfgMap, MettagridCfgFile } from "@/server/api";

export const MapFromCfg: FC<{ cfg: MettagridCfgFile }> = async ({ cfg }) => {
  const maybeMap = await getMettagridCfgMap(cfg.metadata.path);
  if (maybeMap.type === "error") {
    return <div className="text-red-500">Error: {maybeMap.error}</div>;
  }
  return <ExtendedMapViewer mapData={maybeMap.data} />;
};
