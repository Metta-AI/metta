"use client";
import { useQueryState } from "nuqs";
import { type FC, useMemo } from "react";

import { FilterInput } from "@/components/FilterInput";
import { NoResultsMessage } from "@/components/NoResultsMessage";
import { StyledLink } from "@/components/StyledLink";
import { viewMissionRoute } from "@/lib/routes";
import { MissionWithFullConfig } from "@/lib/api/cogames";

export const MissionsRenderer: FC<{
  initialMissions: MissionWithFullConfig[];
}> = ({ initialMissions }) => {
  const [filter, setFilter] = useQueryState("q", { defaultValue: "" });

  const filtered = useMemo(() => {
    if (!filter) return initialMissions;

    return initialMissions.filter(
      (mission) =>
        mission.mission.name.toLowerCase().includes(filter.toLowerCase()) ||
        mission.mission.description
          .toLowerCase()
          .includes(filter.toLowerCase()) ||
        mission.mission.site.name.toLowerCase().includes(filter.toLowerCase())
    );
  }, [initialMissions, filter]);

  return (
    <>
      <div className="mb-4">
        <FilterInput
          className="md:w-1/2 lg:w-1/3"
          placeholder="Filter missions..."
          value={filter}
          onChange={setFilter}
        />
      </div>

      <NoResultsMessage show={filtered.length === 0} />

      <ul className="space-y-2">
        {filtered.map((mission) => {
          const name = `${mission.mission.site.name}.${mission.mission.name}`;

          return (
            <li key={name}>
              <StyledLink href={viewMissionRoute(name)}>{name}</StyledLink>
              <div className="text-sm text-gray-500">
                {mission.mission.description}
              </div>
            </li>
          );
        })}
      </ul>
    </>
  );
};
