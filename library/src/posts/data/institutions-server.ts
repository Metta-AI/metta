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
  // Get all institutions with their papers via the join table
  const institutions = await prisma.institution.findMany({
    include: {
      papers: {
        include: {
          paper: {
            include: {
              paperAuthors: {
                include: {
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
          },
        },
      },
    },
  });

  // Map institutions to DTO format
  const institutionDTOs: InstitutionDTO[] = institutions.map((institution) => {
    const papers = institution.papers.map((pi) => pi.paper);

    // Track authors and their paper counts
    const authorDetails = new Map<
      string,
      { id: string; name: string; paperCount: number }
    >();
    const authorSet = new Set<string>();

    papers.forEach((paper) => {
      paper.paperAuthors.forEach((pa) => {
        authorSet.add(pa.author.id);

        if (authorDetails.has(pa.author.id)) {
          authorDetails.get(pa.author.id)!.paperCount++;
        } else {
          authorDetails.set(pa.author.id, {
            id: pa.author.id,
            name: pa.author.name,
            paperCount: 1,
          });
        }
      });
    });

    // Sort papers by date
    const sortedPapers = papers.sort(
      (a, b) => b.createdAt.getTime() - a.createdAt.getTime()
    );

    // Calculate stars
    const totalUserStars = papers.reduce(
      (sum, paper) => sum + paper.userPaperInteractions.length,
      0
    );
    const avgStars = papers.length > 0 ? totalUserStars / papers.length : 0;

    // Get top categories from tags
    const categoryCount = new Map<string, number>();
    papers.forEach((paper) => {
      paper.tags.forEach((tag) => {
        categoryCount.set(tag, (categoryCount.get(tag) || 0) + 1);
      });
    });

    const topCategories = Array.from(categoryCount.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([category]) => category);

    // Get recent papers
    const recentPapers = sortedPapers.slice(0, 5).map((paper) => ({
      id: paper.id,
      title: paper.title,
      link: paper.link,
      createdAt: paper.createdAt,
      stars: paper.stars,
      authors: paper.paperAuthors.map((pa) => ({
        id: pa.author.id,
        name: pa.author.name,
      })),
    }));

    // Get top authors
    const authors = Array.from(authorDetails.values())
      .sort((a, b) => b.paperCount - a.paperCount)
      .slice(0, 10);

    return {
      name: institution.name,
      paperCount: papers.length,
      authorCount: authorSet.size,
      totalStars: totalUserStars,
      avgStars: Math.round(avgStars * 10) / 10,
      recentActivity:
        sortedPapers.length > 0 ? sortedPapers[0].createdAt : null,
      topCategories,
      recentPapers,
      authors,
    };
  });

  // Sort by paper count (most active institutions first)
  return institutionDTOs.sort((a, b) => b.paperCount - a.paperCount);
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
