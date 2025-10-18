import { fetchJson } from "@/lib/api/client";

export interface PaperSummary {
  id: string;
  title: string;
  createdAt: string;
  stars: number;
  citationCount: number;
  abstractSummary: string | null;
  authors: Array<{
    id: string;
    name: string;
  }>;
  tags: string[];
}

export interface PaperDetail extends PaperSummary {
  abstract: string | null;
  link: string | null;
  source: string | null;
  doi: string | null;
  arxivUrl: string | null;
}

export interface PaperListResponse {
  papers: PaperSummary[];
  pagination: {
    limit: number;
    offset: number;
    total: number;
  };
}

export async function listPapers(query?: {
  limit?: number;
  offset?: number;
  search?: string;
  institution?: string;
  authorId?: string;
}): Promise<PaperListResponse> {
  const params = new URLSearchParams();
  if (query?.limit) params.set("limit", String(query.limit));
  if (query?.offset) params.set("offset", String(query.offset));
  if (query?.search) params.set("search", query.search);
  if (query?.institution) params.set("institution", query.institution);
  if (query?.authorId) params.set("authorId", query.authorId);

  const url =
    params.size > 0 ? `/api/papers?${params.toString()}` : "/api/papers";
  return fetchJson<PaperListResponse>(url);
}

export async function getPaper(id: string): Promise<PaperDetail> {
  return fetchJson<PaperDetail>(`/api/papers/${id}`);
}

export interface PaperInstitutionsResponse {
  hasInstitutions: boolean;
}

export interface PaperDataResponse {
  institutions: string[];
}

export async function checkPaperInstitutions(
  postId: string
): Promise<PaperInstitutionsResponse> {
  return fetchJson<PaperInstitutionsResponse>(
    `/api/papers/${postId}/institutions`
  );
}

export async function getPaperData(postId: string): Promise<PaperDataResponse> {
  return fetchJson<PaperDataResponse>(`/api/papers/${postId}/data`);
}
