import { prisma } from "@/lib/db/prisma";
import { auth } from "@/lib/auth";
import type {
  Institution,
  InstitutionType,
  Paper,
  PaperAuthor,
  PaperInstitution,
  UserInstitution,
} from "@prisma/client";

export type UnifiedInstitutionDTO = {
  id: string;
  name: string;
  domain: string | null;
  description: string | null;
  website: string | null;
  location: string | null;
  type: string;
  isVerified: boolean;
  requiresApproval?: boolean;
  createdAt: Date;
  memberCount: number;
  paperCount: number;
  authorCount: number;
  totalStars: number;
  avgStars: number;
  recentActivity: Date | null;
  topCategories: string[];
  currentUserRole: string | null;
  currentUserStatus: string | null; // APPROVED, PENDING, REJECTED, or null
  members?: Array<{
    id: string;
    user: {
      name: string | null;
      email: string | null;
    };
    role: string | null;
    department: string | null;
    title: string | null;
    joinedAt: Date;
    isActive: boolean;
  }>;
  recentPapers?: Array<{
    id: string;
    title: string;
    abstract: string | null;
    link: string | null;
    createdAt: Date;
    stars: number;
    tags: string[];
    authors: Array<{
      id: string;
      name: string;
    }>;
    institutions: string[];
  }>;
  authors?: Array<{
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
    orcid: string | null;
    googleScholarId: string | null;
    arxivId: string | null;
    recentActivity: Date | null;
    paperCount: number;
    recentPapers: Array<{
      id: string;
      title: string;
      link: string | null;
      createdAt: Date;
      stars: number;
      abstract: string | null;
      tags: string[];
      authors: Array<{
        id: string;
        name: string;
      }>;
      institutions: string[];
    }>;
  }>;
};

type InstitutionWithRelations = Institution & {
  userInstitutions: Array<
    UserInstitution & {
      user: {
        name: string | null;
        email: string | null;
      };
    }
  >;
  papers: Array<
    PaperInstitution & {
      paper: Paper & {
        paperAuthors: Array<
          PaperAuthor & {
            author: { id: string; name: string };
          }
        >;
      };
    }
  >;
  authors: Array<{
    id: string;
    name: string;
    paperAuthors: PaperAuthor[];
  }>;
};

function mapToUnifiedInstitution(
  institution: InstitutionWithRelations,
  options: {
    sessionUserId?: string;
    includeMembers: boolean;
    includePending?: boolean;
  }
): UnifiedInstitutionDTO {
  const { sessionUserId, includeMembers, includePending = false } = options;

  const papers = institution.papers.map((pi) => pi.paper);
  const totalStars = papers.reduce((sum, paper) => sum + paper.stars, 0);
  const avgStars = papers.length > 0 ? totalStars / papers.length : 0;
  const recentActivity = papers.length > 0 ? papers[0].createdAt : null;
  const topCategories = Array.from(
    new Set(papers.flatMap((paper) => paper.tags ?? []))
  )
    .filter((category): category is string => typeof category === "string")
    .slice(0, 5);

  const membership = sessionUserId
    ? institution.userInstitutions.find((ui) => ui.userId === sessionUserId)
    : null;

  const filteredUserInstitutions = institution.userInstitutions.filter((ui) =>
    includePending ? true : ui.status === "APPROVED" && ui.isActive
  );

  return {
    id: institution.id,
    name: institution.name,
    domain: institution.domain,
    description: institution.description,
    website: institution.website,
    location: institution.location,
    type: institution.type,
    isVerified: institution.isVerified,
    requiresApproval: institution.requiresApproval,
    createdAt: institution.createdAt,
    memberCount: filteredUserInstitutions.filter(
      (ui) => ui.status === "APPROVED" && ui.isActive
    ).length,
    paperCount: institution.papers.length,
    authorCount: institution.authors.length,
    totalStars,
    avgStars,
    recentActivity,
    topCategories,
    currentUserRole:
      membership?.status === "APPROVED" ? (membership?.role ?? null) : null,
    currentUserStatus: membership?.status ?? null,
    ...(includeMembers && {
      members: filteredUserInstitutions.map((ui) => ({
        id: ui.id,
        user: ui.user,
        role: ui.role,
        department: ui.department,
        title: ui.title,
        joinedAt: ui.joinedAt,
        isActive: ui.isActive,
      })),
    }),
    recentPapers: papers.map((paper) => ({
      id: paper.id,
      title: paper.title,
      abstract: paper.abstract,
      link: paper.link,
      createdAt: paper.createdAt,
      stars: paper.stars,
      tags: (paper.tags as string[]) || [],
      authors: paper.paperAuthors.map((pa) => ({
        id: pa.author.id,
        name: pa.author.name,
      })),
      institutions:
        "paperInstitutions" in paper
          ? paper.paperInstitutions.map((pi: any) => pi.institution.name)
          : [],
    })),
    authors: institution.authors.map((author) => ({
      id: author.id,
      name: author.name,
      username: author.username,
      email: author.email,
      avatar: author.avatar,
      institution: author.institution,
      department: author.department,
      title: author.title,
      expertise: (author.expertise as string[]) || [],
      hIndex: author.hIndex,
      totalCitations: author.totalCitations,
      claimed: author.claimed,
      orcid: author.orcid,
      googleScholarId: author.googleScholarId,
      arxivId: author.arxivId,
      recentActivity: author.recentActivity,
      paperCount: author.paperAuthors.length,
      recentPapers: author.paperAuthors.slice(0, 5).map((pa) => ({
        id: pa.paper.id,
        title: pa.paper.title,
        abstract: pa.paper.abstract,
        link: pa.paper.link,
        createdAt: pa.paper.createdAt,
        stars: pa.paper.stars,
        tags: (pa.paper.tags as string[]) || [],
        authors: pa.paper.paperAuthors.map((paperAuthor) => ({
          id: paperAuthor.author.id,
          name: paperAuthor.author.name,
        })),
        institutions: pa.paper.paperInstitutions.map(
          (pi) => pi.institution.name
        ),
      })),
    })),
  };
}

/**
 * Load institutions where current user is a member (with full details)
 */
export async function loadUserInstitutions(): Promise<UnifiedInstitutionDTO[]> {
  const session = await auth();

  if (!session?.user?.id) {
    return [];
  }

  const institutions = await prisma.institution.findMany({
    where: {
      userInstitutions: {
        some: {
          userId: session.user.id,
          status: "APPROVED",
          isActive: true,
        },
      },
    },
    include: {
      userInstitutions: {
        where: { status: "APPROVED", isActive: true },
        include: {
          user: {
            select: {
              name: true,
              email: true,
            },
          },
        },
        orderBy: [
          { role: "asc" }, // admins first
          { joinedAt: "asc" },
        ],
      },
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
            },
          },
        },
        orderBy: {
          paper: { createdAt: "desc" },
        },
        take: 5,
      },
      authors: {
        include: {
          paperAuthors: true,
        },
        orderBy: [{ hIndex: "desc" }, { totalCitations: "desc" }],
        take: 10,
      },
    },
    orderBy: { createdAt: "desc" },
  });

  return institutions.map((institution) =>
    mapToUnifiedInstitution(institution, {
      sessionUserId: session.user?.id,
      includeMembers: true,
    })
  );
}

/**
 * Load all institutions (unified view for public directory)
 */
export async function loadAllInstitutions(): Promise<UnifiedInstitutionDTO[]> {
  const session = await auth();

  const institutions = await prisma.institution.findMany({
    include: {
      userInstitutions: {
        where: {
          OR: [
            // Include all approved active memberships
            {
              status: "APPROVED" as const,
              isActive: true,
            },
            // Include current user's pending requests so they can see their status
            ...(session?.user?.id
              ? [
                  {
                    userId: session.user.id,
                    status: "PENDING" as const,
                  },
                ]
              : []),
          ],
        },
        include: {
          user: {
            select: {
              name: true,
              email: true,
            },
          },
        },
      },
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
            },
          },
        },
        orderBy: {
          paper: { createdAt: "desc" },
        },
        take: 5,
      },
      authors: {
        include: {
          paperAuthors: true,
        },
        orderBy: [{ hIndex: "desc" }, { totalCitations: "desc" }],
        take: 10,
      },
    },
    orderBy: [{ isVerified: "desc" }, { createdAt: "desc" }],
  });

  return institutions.map((institution) =>
    mapToUnifiedInstitution(institution, {
      sessionUserId: session?.user?.id,
      includeMembers: false,
      includePending: true,
    })
  );
}

/**
 * Load a single institution by name (unified view for overlay/detail)
 */
export async function loadInstitutionByName(
  name: string
): Promise<UnifiedInstitutionDTO | null> {
  const session = await auth();

  const institution = await prisma.institution.findUnique({
    where: { name },
    include: {
      userInstitutions: {
        where: {
          OR: [
            // Include all approved active memberships
            {
              status: "APPROVED" as const,
              isActive: true,
            },
            // Include current user's pending requests so they can see their status
            ...(session?.user?.id
              ? [
                  {
                    userId: session.user.id,
                    status: "PENDING" as const,
                  },
                ]
              : []),
          ],
        },
        include: {
          user: {
            select: {
              name: true,
              email: true,
            },
          },
        },
      },
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
              paperInstitutions: {
                include: {
                  institution: {
                    select: {
                      name: true,
                    },
                  },
                },
              },
            },
          },
        },
        orderBy: {
          paper: { createdAt: "desc" },
        },
        take: 10, // Get more papers for detail view
      },
      authors: {
        include: {
          paperAuthors: {
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
                  paperInstitutions: {
                    include: {
                      institution: {
                        select: {
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
            take: 5, // Get recent papers per author
          },
        },
        orderBy: [{ hIndex: "desc" }, { totalCitations: "desc" }],
        take: 20, // Get more authors for detail view
      },
    },
  });

  if (!institution) {
    return null;
  }

  // Derive authors from papers if direct author relationship is empty
  let authorsForInstitution = institution.authors;

  if (authorsForInstitution.length === 0) {
    // Get unique authors from papers affiliated with this institution
    const authorIds = new Set<string>();
    const authorMap = new Map<
      string,
      { id: string; name: string; paperCount: number }
    >();

    for (const paperInst of institution.papers) {
      for (const paperAuthor of paperInst.paper.paperAuthors) {
        const authorId = paperAuthor.author.id;
        if (!authorIds.has(authorId)) {
          authorIds.add(authorId);
          authorMap.set(authorId, {
            id: paperAuthor.author.id,
            name: paperAuthor.author.name,
            paperCount: 1,
          });
        } else {
          const existing = authorMap.get(authorId)!;
          existing.paperCount++;
        }
      }
    }

    // Fetch full author data for these authors
    if (authorIds.size > 0) {
      authorsForInstitution = await prisma.author.findMany({
        where: {
          id: { in: Array.from(authorIds) },
        },
        include: {
          paperAuthors: {
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
                  paperInstitutions: {
                    include: {
                      institution: {
                        select: {
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
            take: 5, // Get recent papers per author
          },
        },
        orderBy: [{ hIndex: "desc" }, { totalCitations: "desc" }],
        take: 20,
      });
    }
  }

  return mapToUnifiedInstitution(
    { ...institution, authors: authorsForInstitution },
    {
      sessionUserId: session?.user?.id,
      includeMembers: false,
      includePending: true,
    }
  );
}

// Legacy compatibility exports
export type ManagedInstitutionDTO = UnifiedInstitutionDTO;
export const loadManagedInstitutions = loadUserInstitutions;
export const loadAllManagedInstitutions = loadAllInstitutions;
