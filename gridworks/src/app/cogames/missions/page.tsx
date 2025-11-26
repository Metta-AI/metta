import { getMissions } from "@/lib/api/cogames";
import { MissionsRenderer } from "./MissionsRenderer";

export default async function CogamesMissionsPage() {
  const missions = await getMissions();
  return (
    <div className="p-4">
      <h1 className="mb-4 text-2xl font-bold">Cogames Missions</h1>
      <MissionsRenderer initialMissions={missions} />
    </div>
  );
}
