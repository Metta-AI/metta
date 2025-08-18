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
              institutions: true,
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

  const mappedAuthors = authors.map((author) => {
    const recentPapers = author.paperAuthors.map((pa) => pa.paper);

    // If author doesn't have institution data, try to derive from their papers
    let derivedInstitution = author.institution;
    if (!derivedInstitution && recentPapers.length > 0) {
      // Find the most common institution from their papers
      const institutionCounts = new Map<string, number>();

      recentPapers.forEach((paper) => {
        if (paper.institutions && paper.institutions.length > 0) {
          paper.institutions.forEach((inst) => {
            institutionCounts.set(inst, (institutionCounts.get(inst) || 0) + 1);
          });
        }
      });

      if (institutionCounts.size > 0) {
        // Use the most frequent institution
        derivedInstitution = Array.from(institutionCounts.entries()).sort(
          (a, b) => b[1] - a[1]
        )[0][0];
      }
    }

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
  });

  return mappedAuthors;
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
              institutions: true,
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
    abstract: pa.paper.abstract,
    institutions: pa.paper.institutions,
    tags: pa.paper.tags,
  }));

  // If author doesn't have institution data, try to derive from their papers
  let derivedInstitution = author.institution;
  if (!derivedInstitution && recentPapers.length > 0) {
    // Find the most common institution from their papers
    const institutionCounts = new Map<string, number>();

    recentPapers.forEach((paper) => {
      if (paper.institutions && paper.institutions.length > 0) {
        paper.institutions.forEach((inst) => {
          institutionCounts.set(inst, (institutionCounts.get(inst) || 0) + 1);
        });
      }
    });

    if (institutionCounts.size > 0) {
      // Use the most frequent institution
      derivedInstitution = Array.from(institutionCounts.entries()).sort(
        (a, b) => b[1] - a[1]
      )[0][0];

      console.log(
        `📍 Derived institution for ${author.name}: ${derivedInstitution} (from ${institutionCounts.get(derivedInstitution)} papers)`
      );
    }
  }

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
