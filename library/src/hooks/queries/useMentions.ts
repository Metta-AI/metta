"use client";

import { useQuery } from "@tanstack/react-query";
import * as mentionsApi from "@/lib/api/resources/mentions";
import type { MentionType } from "@/lib/api/resources/mentions";

export function useMentionSearch(params: {
  query: string;
  type: MentionType;
  institutionName?: string;
}) {
  return useQuery({
    queryKey: ["mentions", "search", params],
    queryFn: () =>
      mentionsApi.searchMentions({
        q: params.query,
        type: params.type,
        institutionName: params.institutionName,
      }),
    enabled: params.query.length > 0,
    staleTime: 30000, // Cache results for 30 seconds
  });
}
