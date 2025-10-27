"use client";

import { useQuery } from "@tanstack/react-query";
import * as papersApi from "@/lib/api/resources/papers";

export function usePapers(query?: {
  limit?: number;
  offset?: number;
  search?: string;
  institution?: string;
  authorId?: string;
}) {
  return useQuery({
    queryKey: ["papers", query],
    queryFn: () => papersApi.listPapers(query),
  });
}

export function usePaper(id: string) {
  return useQuery({
    queryKey: ["papers", id],
    queryFn: () => papersApi.getPaper(id),
    enabled: !!id,
  });
}

export function usePaperInstitutions(postId: string) {
  return useQuery({
    queryKey: ["papers", postId, "institutions"],
    queryFn: () => papersApi.checkPaperInstitutions(postId),
    enabled: !!postId,
  });
}

export function usePaperData(postId: string) {
  return useQuery({
    queryKey: ["papers", postId, "data"],
    queryFn: () => papersApi.getPaperData(postId),
    enabled: !!postId,
  });
}
