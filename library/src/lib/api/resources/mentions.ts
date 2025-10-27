import { fetchJson } from "@/lib/api/client";
import type { MentionType } from "@/lib/mentions";

export type { MentionType };

export interface MentionSuggestion {
  type: MentionType;
  id: string;
  value: string; // What gets inserted (@username, @/groupname, etc.)
  display: string; // What user sees in dropdown
  avatar?: string; // User avatar or group icon
  subtitle?: string; // Additional info (email, member count, etc.)
  memberCount?: number; // For groups
}

export interface MentionSearchResponse {
  suggestions: MentionSuggestion[];
}

export interface MentionSearchParams {
  q: string; // Query string
  type: MentionType;
  domain?: string; // For absolute group searches by domain
  institutionName?: string; // For group searches by institution name
  limit?: number; // Max results (default 10)
}

/**
 * Search for mention suggestions
 *
 * Supports searching for users, institutions, and groups with various filtering options.
 */
export async function searchMentions(
  params: MentionSearchParams
): Promise<MentionSearchResponse> {
  const queryParams = new URLSearchParams();
  queryParams.set("q", params.q);
  queryParams.set("type", params.type);
  if (params.domain) {
    queryParams.set("domain", params.domain);
  }
  if (params.institutionName) {
    queryParams.set("institutionName", params.institutionName);
  }
  if (params.limit) {
    queryParams.set("limit", String(params.limit));
  }

  return fetchJson<MentionSearchResponse>(
    `/api/mentions/search?${queryParams.toString()}`
  );
}
