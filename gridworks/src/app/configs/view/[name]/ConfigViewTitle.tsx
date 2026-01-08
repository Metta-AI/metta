"use client";
import { type FC, useEffect, useState } from "react";

import { EditableNode, EditableSuggestion } from "@/components/EditableNode";
import { listConfigMakers, type MakerConfig } from "@/lib/api";

type Props = {
  maker: MakerConfig["maker"];
};

export const ConfigViewTitle: FC<Props> = ({ maker }) => {
  const [suggestions, setSuggestions] = useState<EditableSuggestion[]>([]);

  useEffect(() => {
    async function fetchConfigs() {
      try {
        const configs = await listConfigMakers();
        if (!configs) return;

        const allSuggestions = Object.values(configs)
          .flatMap((list) => list ?? [])
          .map((config) => ({
            text: config.path,
            href: `/configs/view/${encodeURIComponent(config.path)}`,
          }));

        setSuggestions(allSuggestions);
      } catch (error) {
        console.error("Error fetching config makers:", error);
      }
    }
    fetchConfigs();
  }, []);

  return (
    <h1 className="text-2xl font-bold">
      <span className="inline hover:underline">
        <EditableNode text={maker.path} suggestions={suggestions}>
          <a
            className="hover:underline"
            href={`cursor://file${maker.absolute_path}:${maker.line}`}
          >
            {maker.path}
          </a>
        </EditableNode>
      </span>
      <span className="text-xl text-gray-400"> &rarr;</span>
      <span className="text-xl text-gray-500"> {maker.kind}</span>
    </h1>
  );
};
