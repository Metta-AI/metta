"use client";

import { useQuery } from "@tanstack/react-query";
import * as authorsApi from "@/lib/api/resources/authors";

export function useAuthors(
  params?: { search?: string },
  options?: { enabled?: boolean }
) {
  return useQuery({
    queryKey: ["authors", params],
    queryFn: () => authorsApi.listAuthors(params),
    enabled: options?.enabled ?? true,
  });
}

export function useAuthor(id: string, options?: { enabled?: boolean }) {
  return useQuery({
    queryKey: ["authors", id],
    queryFn: () => authorsApi.getAuthor(id),
    enabled: options?.enabled ?? !!id,
  });
}
