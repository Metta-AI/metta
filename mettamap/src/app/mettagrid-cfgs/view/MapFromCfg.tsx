import { FC } from "react";

import { StorableMapViewer } from "@/components/StorableMapViewer";
import { getMettagridCfgMap, MettagridCfgFile } from "@/lib/api";

export const MapFromCfg: FC<{ cfg: MettagridCfgFile }> = async ({ cfg }) => {
  const maybeMap = await getMettagridCfgMap(cfg.metadata.path);
  if (maybeMap.type === "error") {
    return <div className="text-red-500">Error: {maybeMap.error}</div>;
  }
  return <StorableMapViewer map={maybeMap.data} />;
};
