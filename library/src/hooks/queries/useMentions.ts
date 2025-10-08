"use client";

import { useQuery } from "@tanstack/react-query";
import * as mentionsApi from "@/lib/api/resources/mentions";

export function useMentionSearch(params: {
  query: string;
  institutionName?: string;
}) {
  return useQuery({
    queryKey: ["mentions", "search", params],
    queryFn: () => mentionsApi.searchMentions(params),
    enabled: params.query.length > 0,
    staleTime: 30000, // Cache results for 30 seconds
  });
}
