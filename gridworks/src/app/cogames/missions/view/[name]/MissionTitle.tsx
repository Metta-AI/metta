"use client";
import { type FC, useEffect, useMemo, useState } from "react";

import { EditableNode, EditableSuggestion } from "@/components/EditableNode";
import { getMissions, type MissionWithFullConfig } from "@/lib/api/cogames";

export const MissionTitle: FC<{ mission: MissionWithFullConfig }> = ({
  mission,
}) => {
  const [suggestions, setSuggestions] = useState<EditableSuggestion[]>([]);
  const [filter, setFilter] = useState<string>("");

  useEffect(() => {
    async function fetchMissions() {
      try {
        const missions = await getMissions();
        const allSuggestions: EditableSuggestion[] = missions.map((m) => {
          const text = `${m.mission.site.name}.${m.mission.name}`;
          return {
            text: text,
            href: `/cogames/missions/view/${encodeURIComponent(text)}`,
          };
        });
        setSuggestions(allSuggestions);
      } catch (error) {
        console.error("Error fetching missions:", error);
      }
    }
    fetchMissions();
  }, []);

  const filteredSuggestions = useMemo(() => {
    const currentText = `${mission.mission.site.name}.${mission.mission.name}`;
    if (!filter) return suggestions;
    if (filter === currentText) return suggestions;

    return suggestions.filter((sug) =>
      sug.text.toLowerCase().includes(filter.toLowerCase())
    );
  }, [suggestions, filter, mission.mission.site.name, mission.mission.name]);

  return (
    <h1 className="font-mono text-2xl font-bold text-gray-700">
      <span className="text-gray-500">CoGames Mission: </span>
      <EditableNode
        text={`${mission.mission.site.name}.${mission.mission.name}`}
        suggestions={filteredSuggestions}
        onChange={setFilter}
      >
        {mission.mission.site.name}
        <span className="text-gray-500">.</span>
        {mission.mission.name}
      </EditableNode>
    </h1>
  );
};
