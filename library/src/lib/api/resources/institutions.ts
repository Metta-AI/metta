import { fetchJson } from "@/lib/api/client";

export interface InstitutionAuthor {
  id: string;
  name: string;
  paperCount: number;
}

export interface InstitutionSummary {
  id: string;
  name: string;
  domain: string | null;
  memberCount: number;
  paperCount: number;
  authorCount: number;
  totalStars: number;
  avgStars: number;
  topCategories: string[];
  isVerified: boolean;
  requiresApproval: boolean;
  recentActivity: string | null;
  authors: InstitutionAuthor[];
}

export interface InstitutionDetail extends InstitutionSummary {
  description: string | null;
  website: string | null;
  location: string | null;
  recentPapers: Array<{
    id: string;
    title: string;
    link: string | null;
    createdAt: string;
    stars: number;
    authors: Array<{ id: string; name: string }>;
  }>;
}

export async function listInstitutions(): Promise<InstitutionSummary[]> {
  return fetchJson<InstitutionSummary[]>("/api/institutions");
}

export async function getInstitutionByName(
  name: string
): Promise<InstitutionDetail | null> {
  try {
    return await fetchJson<InstitutionDetail>(
      `/api/institutions/${encodeURIComponent(name)}`
    );
  } catch (error) {
    if (
      error instanceof Error &&
      "status" in error &&
      (error as { status: number }).status === 404
    ) {
      return null;
    }

    throw error;
  }
}
