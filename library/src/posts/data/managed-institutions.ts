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
    link: string | null;
    createdAt: Date;
    stars: number;
    authors: Array<{
      id: string;
      name: string;
    }>;
  }>;
  authors?: Array<{
    id: string;
    name: string;
    paperCount: number;
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
      link: paper.link,
      createdAt: paper.createdAt,
      stars: paper.stars,
      authors: paper.paperAuthors.map((pa) => ({
        id: pa.author.id,
        name: pa.author.name,
      })),
    })),
    authors: institution.authors.map((author) => ({
      id: author.id,
      name: author.name,
      paperCount: author.paperAuthors.length,
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

// Legacy compatibility exports
export type ManagedInstitutionDTO = UnifiedInstitutionDTO;
export const loadManagedInstitutions = loadUserInstitutions;
export const loadAllManagedInstitutions = loadAllInstitutions;
