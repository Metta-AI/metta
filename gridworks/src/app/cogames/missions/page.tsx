import { StyledLink } from "@/components/StyledLink";
import { getMissions } from "@/lib/api/cogames";
import { viewMissionRoute } from "@/lib/routes";

export default async function CogamesMissionsPage() {
  const missions = await getMissions();
  return (
    <div className="p-4">
      <h1 className="mb-4 text-2xl font-bold">Cogames Missions</h1>
      <ul className="space-y-2">
        {missions.map((mission) => {
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
    </div>
  );
}
