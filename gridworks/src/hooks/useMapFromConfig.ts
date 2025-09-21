import { useEffect, useState } from "react";

import { Config, getConfigMap, StorableMap } from "@/lib/api";

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

export function useMapFromConfig(cfg: Config, name?: string) {
  const [id, setId] = useState(0);

  const [map, setMap] = useState<MapState>({ type: "loading" });
  useEffect(() => {
    setMap({ type: "loading" });
    getConfigMap(cfg.maker.path, name)
      .then((map) => setMap({ type: "map", map }))
      .catch((e) => {
        console.error(e);
        setMap({ type: "error", error: e });
      });
  }, [cfg.maker.path, id]);

  const reload = () => {
    setId(id + 1);
  };

  return { map, reload };
}
