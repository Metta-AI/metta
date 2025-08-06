import { prisma } from "@/lib/db/prisma";

export type InstitutionDTO = {
  name: string;
  paperCount: number;
  authorCount: number;
  totalStars: number;
  avgStars: number;
  recentActivity: Date | null;
  topCategories: string[];
  recentPapers: Array<{
    id: string;
    title: string;
    link: string | null;
    createdAt: Date;
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

/**
 * Load all institutions with aggregated statistics from papers
 */
export async function loadInstitutions(): Promise<InstitutionDTO[]> {
  // Get all papers with institutions and their user interactions for star counting
  const papers = await prisma.paper.findMany({
    where: {
      institutions: {
        isEmpty: false,
      },
    },
    select: {
      id: true,
      title: true,
      link: true,
      createdAt: true,
      stars: true,
      institutions: true,
      tags: true,
      paperAuthors: {
        select: {
          author: {
            select: {
              id: true,
              name: true,
            },
          },
        },
      },
      userPaperInteractions: {
        where: {
          starred: true,
        },
        select: {
          starred: true,
        },
      },
    },
    orderBy: {
      createdAt: "desc",
    },
  });

  // Aggregate data by institution
  const institutionMap = new Map<
    string,
    {
      papers: Array<{
        id: string;
        title: string;
        link: string | null;
        createdAt: Date;
        stars: number;
        userStars: number;
        tags: string[];
        authors: Array<{ id: string; name: string }>;
      }>;
      authors: Set<string>;
      authorDetails: Map<
        string,
        { id: string; name: string; paperCount: number }
      >;
    }
  >();

  // Process each paper
  papers.forEach((paper) => {
    const paperAuthors = paper.paperAuthors.map((pa) => pa.author);
    const userStarCount = paper.userPaperInteractions.length; // Count of users who starred this paper

    paper.institutions.forEach((institution) => {
      if (!institutionMap.has(institution)) {
        institutionMap.set(institution, {
          papers: [],
          authors: new Set(),
          authorDetails: new Map(),
        });
      }

      const instData = institutionMap.get(institution)!;

      // Add paper with user star count
      instData.papers.push({
        id: paper.id,
        title: paper.title,
        link: paper.link,
        createdAt: paper.createdAt,
        stars: paper.stars,
        userStars: userStarCount,
        tags: paper.tags,
        authors: paperAuthors,
      });

      // Track authors
      paperAuthors.forEach((author) => {
        instData.authors.add(author.id);

        if (instData.authorDetails.has(author.id)) {
          const authorDetail = instData.authorDetails.get(author.id)!;
          authorDetail.paperCount++;
        } else {
          instData.authorDetails.set(author.id, {
            id: author.id,
            name: author.name,
            paperCount: 1,
          });
        }
      });
    });
  });

  // Convert to InstitutionDTO array
  const institutions: InstitutionDTO[] = Array.from(
    institutionMap.entries()
  ).map(([name, data]) => {
    const sortedPapers = data.papers.sort(
      (a, b) => b.createdAt.getTime() - a.createdAt.getTime()
    );
    const recentPapers = sortedPapers.slice(0, 5);

    const totalUserStars = data.papers.reduce((sum, paper) => sum + paper.userStars, 0);
    const avgStars =
      data.papers.length > 0 ? totalUserStars / data.papers.length : 0;

    // Get top categories from tags
    const categoryCount = new Map<string, number>();
    data.papers.forEach((paper) => {
      paper.tags.forEach((tag) => {
        categoryCount.set(tag, (categoryCount.get(tag) || 0) + 1);
      });
    });

    const topCategories = Array.from(categoryCount.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([category]) => category);

    // Get most recent activity
    const recentActivity =
      sortedPapers.length > 0 ? sortedPapers[0].createdAt : null;

    // Convert authors to array and sort by paper count
    const authors = Array.from(data.authorDetails.values())
      .sort((a, b) => b.paperCount - a.paperCount)
      .slice(0, 10); // Top 10 authors

    return {
      name,
      paperCount: data.papers.length,
      authorCount: data.authors.size,
      totalStars: totalUserStars,
      avgStars: Math.round(avgStars * 10) / 10,
      recentActivity,
      topCategories,
      recentPapers,
      authors,
    };
  });

  // Sort by paper count (most active institutions first)
  return institutions.sort((a, b) => b.paperCount - a.paperCount);
}

/**
 * Load a single institution with detailed information
 */
export async function loadInstitution(
  institutionName: string
): Promise<InstitutionDTO | null> {
  const institutions = await loadInstitutions();
  return institutions.find((inst) => inst.name === institutionName) || null;
}

/**
 * Get institution statistics summary
 */
export async function getInstitutionStats() {
  const institutions = await loadInstitutions();

  return {
    totalInstitutions: institutions.length,
    totalPapers: institutions.reduce((sum, inst) => sum + inst.paperCount, 0),
    totalAuthors: institutions.reduce((sum, inst) => sum + inst.authorCount, 0),
    topInstitutions: institutions.slice(0, 10),
  };
}
