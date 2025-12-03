"use client";
import { type FC, useEffect, useMemo, useState } from "react";

import {
  EditableSuggestion,
  EditableTextNode,
} from "@/components/EditableTextNode";
import { EditIcon } from "@/components/icons/EditIcon";
import { listConfigMakers } from "@/lib/api";

type Props = {
  title: string;
  kind: string;
  filteredConfigs?: EditableSuggestion[];
};

export const ConfigViewTitle: FC<Props> = ({ title, kind }) => {
  const [mode, setMode] = useState<"edit" | "view">("view");
  const [configs, setConfigs] = useState<EditableSuggestion[]>([]);
  const [filter, setFilter] = useState<string>("");

  useEffect(() => {
    async function fetchConfigs() {
      try {
        const cfgs = await listConfigMakers();
        if (!cfgs) return;

        const paths: EditableSuggestion[] = [];
        for (const kind of Object.keys(cfgs)) {
          const cfgList = cfgs[kind as keyof typeof cfgs];
          if (!cfgList) continue;

          for (const cfg of cfgList) {
            paths.push({
              text: cfg.path,
              hyperlink: `/configs/view/${encodeURIComponent(cfg.path)}`,
            });
          }
        }

        setConfigs(paths);
      } catch (error) {
        console.error("Error fetching config makers:", error);
      }
    }
    fetchConfigs();
  }, []);

  const filteredConfigs = useMemo(() => {
    if (!filter) return configs;
    if (filter === title) return configs;

    return configs.filter((cfg) =>
      cfg.text.toLowerCase().includes(filter.toLowerCase())
    );
  }, [configs, filter, title]);

  return (
    <h1 className="text-2xl font-bold">
      <span
        className="mr-2 cursor-pointer rounded pl-1 hover:bg-gray-200"
        onClick={() => setMode((m) => (m === "view" ? "edit" : "view"))}
      >
        <EditIcon className="inline align-middle" />
      </span>
      <span>
        <div className="inline hover:underline">
          <EditableTextNode
            text={title}
            mode={mode}
            onModeChange={(newMode) => setMode(newMode)}
            onChange={setFilter}
            suggestions={filteredConfigs}
          />
        </div>
      </span>
      <span className="text-xl text-gray-400"> &rarr;</span>
      <span className="text-xl text-gray-500"> {kind}</span>
    </h1>
  );
};
