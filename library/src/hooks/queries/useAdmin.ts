"use client";

import { useQuery } from "@tanstack/react-query";
import * as adminApi from "@/lib/api/resources/admin";

export function useAdminInstitutions() {
  return useQuery({
    queryKey: ["admin", "institutions"],
    queryFn: () => adminApi.listAdminInstitutions(),
  });
}

export function useAdminUsers() {
  return useQuery({
    queryKey: ["admin", "users"],
    queryFn: () => adminApi.listAdminUsers(),
  });
}
