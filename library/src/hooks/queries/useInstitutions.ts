"use client";

import { useQuery } from "@tanstack/react-query";
import * as institutionsApi from "@/lib/api/resources/institutions";

export function useInstitutions() {
  return useQuery({
    queryKey: ["institutions"],
    queryFn: () => institutionsApi.listInstitutions(),
  });
}

export function useInstitution(name: string, options?: { enabled?: boolean }) {
  return useQuery({
    queryKey: ["institutions", name],
    queryFn: () => institutionsApi.getInstitutionByName(name),
    enabled: options?.enabled ?? !!name,
  });
}
