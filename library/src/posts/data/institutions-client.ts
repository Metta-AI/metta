// Client-side only institution data types and functions

export type InstitutionDTO = {
  name: string;
  paperCount: number;
  authorCount: number;
  totalStars: number;
  avgStars: number;
  recentActivity: Date | string | null;
  topCategories: string[];
  recentPapers: Array<{
    id: string;
    title: string;
    link: string | null;
    createdAt: Date | string;
    stars: number;
    authors: Array<{
      id: string;
      name: string;
    }>;
  }>;
  authors: Array<{
    id: string;
    name: string;
    paperCount: number;
  }>;
};

// Client-side version that fetches from an API endpoint
export async function loadInstitutionClient(
  institutionName: string
): Promise<InstitutionDTO | null> {
  try {
    const response = await fetch(
      `/api/institutions/${encodeURIComponent(institutionName)}`
    );
    if (!response.ok) {
      return null;
    }
    return await response.json();
  } catch (error) {
    console.error("Error loading institution:", error);
    return null;
  }
}
