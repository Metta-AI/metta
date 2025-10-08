"use client";

import { useQuery } from "@tanstack/react-query";
import * as authorsApi from "@/lib/api/resources/authors";

export function useAuthors(params?: { search?: string }) {
  return useQuery({
    queryKey: ["authors", params],
    queryFn: () => authorsApi.listAuthors(params),
  });
}

export function useAuthor(id: string) {
  return useQuery({
    queryKey: ["authors", id],
    queryFn: () => authorsApi.getAuthor(id),
    enabled: !!id,
  });
}

export function useAuthorSearch(query: string) {
  return useQuery({
    queryKey: ["authors", "search", query],
    queryFn: () => authorsApi.searchAuthors(query),
    enabled: query.length > 0,
  });
}
