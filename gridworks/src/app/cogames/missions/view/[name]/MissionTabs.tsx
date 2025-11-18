"use client";

import { useSelectedLayoutSegment } from "next/navigation";
import { FC, PropsWithChildren } from "react";

import { ErrorBoundary } from "@/components/ErrorBoundary";
import { LinkTabs } from "@/components/LinkTabs";
import { Variant } from "@/lib/api/cogames";
import {
  viewMissionEnvRoute,
  viewMissionMapRoute,
  viewMissionRoute,
} from "@/lib/routes";

import { useVariants } from "./useVariants";
import { VariantsSelector } from "./VariantsSelector";

export const MissionTabs: FC<
  PropsWithChildren<{ name: string; allVariants: Variant[] }>
> = ({ name, allVariants, children }) => {
  const segment = useSelectedLayoutSegment() ?? "config";
  const [variants] = useVariants();

  return (
    <LinkTabs
      tabs={[
        {
          id: "config",
          label: "Mission Config",
          href: viewMissionRoute(name, variants),
          isActive: segment === null,
        },
        {
          id: "env",
          label: "MettaGridEnv Config",
          href: viewMissionEnvRoute(name, variants),
        },
        {
          id: "map",
          label: "Map",
          href: viewMissionMapRoute(name, variants),
        },
      ]}
      activeTab={segment}
      additionalTabBarContent={<VariantsSelector allVariants={allVariants} />}
    >
      <ErrorBoundary resetKeys={[variants.join(",")]}>{children}</ErrorBoundary>
    </LinkTabs>
  );
};
