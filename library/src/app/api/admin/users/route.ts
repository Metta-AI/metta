import { NextRequest, NextResponse } from "next/server";

import { getAdminSessionOrRedirect } from "@/lib/adminAuth";
import { prisma } from "@/lib/db/prisma";
import { withErrorHandler } from "@/lib/api/error-handler";

/**
 * GET /api/admin/users
 * Get all users for admin management
 */
export const GET = withErrorHandler(async (request: NextRequest) => {
  await getAdminSessionOrRedirect();

  const { searchParams } = new URL(request.url);
  const page = parseInt(searchParams.get("page") || "1", 10);
  const limit = Math.min(parseInt(searchParams.get("limit") || "20", 10), 100); // Cap at 100 to prevent abuse
  const search = searchParams.get("search") || "";
  const showBannedOnly = searchParams.get("banned") === "true";

  const skip = (page - 1) * limit;

  // Build where clause for filtering
  const whereClause: any = {
    // Exclude system bot user
    email: { not: "library_bot@system" },
  };

  if (showBannedOnly) {
    whereClause.isBanned = true;
  }

  if (search) {
    whereClause.OR = [
      { name: { contains: search, mode: "insensitive" } },
      { email: { contains: search, mode: "insensitive" } },
    ];
  }

  const [users, totalCount] = await Promise.all([
    prisma.user.findMany({
      where: whereClause,
      select: {
        id: true,
        name: true,
        email: true,
        emailVerified: true,
        image: true,
        isBanned: true,
        bannedAt: true,
        banReason: true,
        bannedBy: {
          select: {
            id: true,
            name: true,
            email: true,
          },
        },
        _count: {
          select: {
            posts: true,
            comments: true,
            userInstitutions: {
              where: { status: "APPROVED", isActive: true },
            },
          },
        },
      },
      orderBy: [
        { isBanned: "desc" }, // Banned users first
        { email: "asc" },
      ],
      skip,
      take: limit,
    }),

    prisma.user.count({ where: whereClause }),
  ]);

  const totalPages = Math.ceil(totalCount / limit);

  return NextResponse.json({
    users: users.map((user) => ({
      id: user.id,
      name: user.name,
      email: user.email,
      emailVerified: user.emailVerified,
      image: user.image,
      isBanned: user.isBanned,
      bannedAt: user.bannedAt,
      banReason: user.banReason,
      bannedBy: user.bannedBy,
      postCount: user._count.posts,
      commentCount: user._count.comments,
      institutionCount: user._count.userInstitutions,
    })),
    pagination: {
      page,
      limit,
      total: totalCount,
      totalPages,
      hasNextPage: page < totalPages,
      hasPrevPage: page > 1,
    },
  });
});
