import { useCallback, useEffect, useState } from "react";

import { getConfigMap, MakerConfig, StorableMap } from "@/lib/api";
import { getMissionMap } from "@/lib/api/cogames";

export type MapState =
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

export function useLoadMapFromConfig(cfg: MakerConfig, name?: string) {
  const [state, setState] = useState<MapState>({ type: "init", loading: true });

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

export function useLoadMapFromMission(name: string, variants: string[]) {
  const [state, setState] = useState<MapState>({ type: "init", loading: true });

  const reload = useCallback(async () => {
    setState((state) => ({
      ...state,
      loading: true,
    }));

    try {
      const map = await getMissionMap(name, variants);
      setState({ type: "map", map, loading: false });
    } catch (e) {
      console.error(e);
      setState({ type: "error", error: e as Error, loading: false });
    }
  }, [name, variants]);

  useEffect(() => {
    reload();
  }, [reload]);

  return { mapState: state, reload };
}
