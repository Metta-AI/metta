// Client-side only author data types and functions

export type AuthorDTO = {
  id: string;
  name: string;
  username: string | null;
  email: string | null;
  avatar: string | null;
  institution: string | null;
  department: string | null;
  title: string | null;
  expertise: string[];
  hIndex: number | null;
  totalCitations: number | null;
  claimed: boolean;
  isFollowing: boolean;
  recentActivity: Date | string | null;
  orcid: string | null;
  googleScholarId: string | null;
  arxivId: string | null;
  createdAt: Date | string;
  updatedAt: Date | string;
  paperCount: number;
  recentPapers: Array<{
    id: string;
    title: string;
    link: string | null;
    createdAt: Date | string;
    stars: number;
  }>;
};

// Client-side version that fetches from an API endpoint
export async function loadAuthorClient(authorId: string): Promise<AuthorDTO | null> {
  try {
    const response = await fetch(`/api/authors/${authorId}`);
    if (!response.ok) {
      return null;
    }
    return await response.json();
  } catch (error) {
    console.error('Error loading author:', error);
    return null;
  }
} 