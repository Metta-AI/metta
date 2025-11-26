"use client";
import { type FC, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import { StyledLink } from "@/components/StyledLink";
import { FilterInput } from "@/components/FilterInput";
import { NoResultsMessage } from "@/components/NoResultsMessage";
import { viewMissionRoute } from "@/lib/routes";
import { MissionWithFullConfig } from "@/lib/api/cogames";

export const MissionsRenderer: FC<{
  initialMissions: MissionWithFullConfig[];
}> = ({ initialMissions }) => {
  const searchParams = useSearchParams();
  // QoL, share links with pre-filled filter.
  const initialFilter = searchParams.get("q") || "";
  const [filter, setFilter] = useState(initialFilter);

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
          value={filter}
          onChange={setFilter}
          placeholder="Filter missions..."
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
