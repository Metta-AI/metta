import { prisma } from "@/lib/db/prisma";

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
  recentActivity: Date | null;
  orcid: string | null;
  googleScholarId: string | null;
  arxivId: string | null;
  createdAt: Date;
  updatedAt: Date;
  paperCount: number;
  recentPapers: Array<{
    id: string;
    title: string;
    link: string | null;
    createdAt: Date;
    stars: number;
  }>;
};

/**
 * Helper: Derive institution from author's papers
 * Finds the most common institution across an author's papers
 */
function deriveInstitutionFromPapers(
  papers: Array<{
    paperInstitutions?: Array<{ institution: { name: string } }>;
  }>,
  authorInstitution: string | null
): string | null {
  if (authorInstitution) {
    return authorInstitution;
  }

  if (papers.length === 0) {
    return null;
  }

  const institutionCounts = new Map<string, number>();

  papers.forEach((paper) => {
    if (paper.paperInstitutions && paper.paperInstitutions.length > 0) {
      paper.paperInstitutions.forEach((pi: any) => {
        const name = pi.institution.name;
        institutionCounts.set(name, (institutionCounts.get(name) || 0) + 1);
      });
    }
  });

  if (institutionCounts.size === 0) {
    return null;
  }

  // Return the most frequent institution
  return Array.from(institutionCounts.entries()).sort(
    (a, b) => b[1] - a[1]
  )[0][0];
}

/**
 * Helper: Map author data to AuthorDTO
 */
function mapToAuthorDTO(
  author: {
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
    recentActivity: Date | null;
    orcid: string | null;
    googleScholarId: string | null;
    arxivId: string | null;
    createdAt: Date;
    updatedAt: Date;
    paperAuthors: any[];
  },
  recentPapers: any[],
  derivedInstitution: string | null
): AuthorDTO {
  return {
    id: author.id,
    name: author.name,
    username: author.username,
    email: author.email,
    avatar: author.avatar,
    institution: derivedInstitution,
    department: author.department,
    title: author.title,
    expertise: author.expertise,
    hIndex: author.hIndex,
    totalCitations: author.totalCitations,
    claimed: author.claimed,
    isFollowing: false, // Default to not following for now
    recentActivity: author.recentActivity,
    orcid: author.orcid,
    googleScholarId: author.googleScholarId,
    arxivId: author.arxivId,
    createdAt: author.createdAt,
    updatedAt: author.updatedAt,
    paperCount: author.paperAuthors.length,
    recentPapers,
  };
}

export async function loadAuthors(): Promise<AuthorDTO[]> {
  const authors = await prisma.author.findMany({
    include: {
      paperAuthors: {
        include: {
          paper: {
            select: {
              id: true,
              title: true,
              link: true,
              createdAt: true,
              stars: true,
              paperInstitutions: {
                select: {
                  institution: {
                    select: {
                      id: true,
                      name: true,
                    },
                  },
                },
              },
            },
          },
        },
        orderBy: {
          paper: {
            createdAt: "desc",
          },
        },
        take: 5, // Get only the 5 most recent papers for performance
      },
    },
    orderBy: {
      name: "asc",
    },
  });

  return authors.map((author) => {
    const recentPapers = author.paperAuthors.map((pa) => pa.paper);
    const derivedInstitution = deriveInstitutionFromPapers(
      recentPapers,
      author.institution
    );
    return mapToAuthorDTO(author, recentPapers, derivedInstitution);
  });
}

export async function loadAuthor(authorId: string): Promise<AuthorDTO | null> {
  const author = await prisma.author.findUnique({
    where: { id: authorId },
    include: {
      paperAuthors: {
        include: {
          paper: {
            select: {
              id: true,
              title: true,
              link: true,
              createdAt: true,
              stars: true,
              abstract: true,
              paperInstitutions: {
                select: {
                  institution: {
                    select: {
                      id: true,
                      name: true,
                    },
                  },
                },
              },
              tags: true,
            },
          },
        },
        orderBy: {
          paper: {
            createdAt: "desc",
          },
        },
      },
    },
  });

  if (!author) {
    return null;
  }

  const recentPapers = author.paperAuthors.map((pa) => ({
    id: pa.paper.id,
    title: pa.paper.title,
    link: pa.paper.link,
    createdAt: pa.paper.createdAt,
    stars: pa.paper.stars,
  }));

  const derivedInstitution = deriveInstitutionFromPapers(
    author.paperAuthors.map((pa) => pa.paper),
    author.institution
  );

  return mapToAuthorDTO(author, recentPapers, derivedInstitution);
}
