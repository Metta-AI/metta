import { PropsWithChildren } from "react";

import { StyledLink } from "@/components/StyledLink";
import { getMission, getVariants } from "@/lib/api/cogames";
import { cogamesMissionsRoute } from "@/lib/routes";

import { MissionTabs } from "./MissionTabs";

export default async function ViewMissionLayout({
  params,
  children,
}: PropsWithChildren<{
  params: Promise<{ name: string }>;
}>) {
  const { name } = await params;

  // load without variants - should be good enough (unless variants decide to edit mission description...)
  const mission = await getMission(name);
  const allVariants = await getVariants();

  return (
    <div>
      <div className="border-b border-gray-200 bg-gray-100 p-4">
        <div>
          <StyledLink href={cogamesMissionsRoute()} className="text-sm">
            ← Back to missions list
          </StyledLink>
        </div>
        <h1 className="font-mono text-2xl font-bold text-gray-700">
          <span className="text-gray-500">CoGames Mission: </span>
          {mission.mission.site.name}
          <span className="text-gray-500">.</span>
          {mission.mission.name}
        </h1>
        <div className="mt-1 text-sm text-gray-500">
          {mission.mission.description}
        </div>
      </div>
      <div className="p-4">
        <MissionTabs name={name} allVariants={allVariants}>
          {children}
        </MissionTabs>
      </div>
    </div>
  );
}
