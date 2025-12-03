"use client";
import { type FC, useEffect, useMemo, useState } from "react";

import {
  EditableSuggestion,
  EditableTextNode,
} from "@/components/EditableTextNode";
import { EditIcon } from "@/components/icons/EditIcon";
import { listConfigMakers, type MakerConfig } from "@/lib/api";

type Props = {
  maker: MakerConfig["maker"];
};

export const ConfigViewTitle: FC<Props> = ({ maker }) => {
  const [mode, setMode] = useState<"edit" | "view">("view");
  const [suggestions, setSuggestions] = useState<EditableSuggestion[]>([]);
  const [filter, setFilter] = useState<string>("");

  useEffect(() => {
    async function fetchConfigs() {
      try {
        const cfgs = await listConfigMakers();
        if (!cfgs) return;

        const allSuggestions: EditableSuggestion[] = [];
        for (const kind of Object.keys(cfgs)) {
          const cfgList = cfgs[kind as keyof typeof cfgs];
          if (!cfgList) continue;

          for (const cfg of cfgList) {
            allSuggestions.push({
              text: cfg.path,
              hyperlink: `/configs/view/${encodeURIComponent(cfg.path)}`,
            });
          }
        }

        setSuggestions(allSuggestions);
      } catch (error) {
        console.error("Error fetching config makers:", error);
      }
    }
    fetchConfigs();
  }, []);

  const filteredSuggestions = useMemo(() => {
    if (!filter) return suggestions;
    if (filter === maker.path) return suggestions;

    return suggestions.filter((sug) =>
      sug.text.toLowerCase().includes(filter.toLowerCase())
    );
  }, [suggestions, filter, maker.path]);

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
            text={maker.path}
            mode={mode}
            href={`cursor://file${maker.absolute_path}:${maker.line}`}
            onModeChange={(newMode) => setMode(newMode)}
            onChange={setFilter}
            suggestions={filteredSuggestions}
          />
        </div>
      </span>
      <span className="text-xl text-gray-400"> &rarr;</span>
      <span className="text-xl text-gray-500"> {maker.kind}</span>
    </h1>
  );
};
