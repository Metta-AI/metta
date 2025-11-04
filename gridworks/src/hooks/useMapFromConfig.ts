import { useCallback, useEffect, useState } from "react";

import { Config, getConfigMap, StorableMap } from "@/lib/api";

type MapState =
  | {
      type: "init";
      loading: true;
    }
  | {
      type: "error";
      error: Error;
      loading: boolean;
    }
  | {
      type: "map";
      map: StorableMap;
      loading: boolean;
    };

export function useMapFromConfig(cfg: Config, name?: string) {
  const [state, setState] = useState<MapState>({ type: "loading" });

  const reload = useCallback(async () => {
    setState((state) => ({
      ...state,
      loading: true,
    }));

    try {
      const map = await getConfigMap(cfg.maker.path, name);
      setState({ type: "map", map, loading: false });
    } catch (e) {
      console.error(e);
      setState({ type: "error", error: e as Error, loading: false });
    }
  }, [cfg.maker.path, name]);

  useEffect(() => {
    reload();
  }, [reload]);

  return { mapState: state, reload };
}
