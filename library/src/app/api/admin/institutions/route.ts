import { NextRequest, NextResponse } from "next/server";
import { z } from "zod/v4";

import { getAdminSessionOrRedirect } from "@/lib/adminAuth";
import { prisma } from "@/lib/db/prisma";
import { withErrorHandler } from "@/lib/api/error-handler";

/**
 * GET /api/admin/institutions
 * Get all institutions for admin management
 */
export const GET = withErrorHandler(async (request: NextRequest) => {
  await getAdminSessionOrRedirect();

  const institutions = await prisma.institution.findMany({
    select: {
      id: true,
      name: true,
      domain: true,
      type: true,
      isVerified: true,
      requiresApproval: true,
      createdAt: true,
      userInstitutions: {
        where: {
          role: "admin",
          status: "APPROVED",
          isActive: true,
        },
        select: {
          id: true,
          role: true,
          joinedAt: true,
          department: true,
          title: true,
          user: {
            select: {
              id: true,
              name: true,
              email: true,
            },
          },
        },
        orderBy: [{ joinedAt: "asc" }],
      },
      _count: {
        select: {
          userInstitutions: {
            where: { status: "APPROVED", isActive: true },
          },
        },
      },
    },
    orderBy: [{ name: "asc" }],
  });

  // Get pending requests for all institutions
  const allPendingRequests = await prisma.userInstitution.findMany({
    where: {
      status: "PENDING",
      institutionId: { in: institutions.map((i) => i.id) },
    },
    select: {
      id: true,
      institutionId: true,
      joinedAt: true,
      role: true,
      department: true,
      title: true,
      user: {
        select: {
          id: true,
          name: true,
          email: true,
        },
      },
    },
    orderBy: [{ joinedAt: "asc" }],
  });

  // Group pending requests by institution
  const pendingByInstitution = allPendingRequests.reduce(
    (acc, request) => {
      if (!acc[request.institutionId]) {
        acc[request.institutionId] = [];
      }
      acc[request.institutionId].push(request);
      return acc;
    },
    {} as Record<string, typeof allPendingRequests>
  );

  return NextResponse.json({
    institutions: institutions.map((institution) => ({
      id: institution.id,
      name: institution.name,
      domain: institution.domain,
      type: institution.type,
      isVerified: institution.isVerified,
      requiresApproval: institution.requiresApproval,
      createdAt: institution.createdAt,
      memberCount: institution._count.userInstitutions,
      pendingCount: pendingByInstitution[institution.id]?.length || 0,
      admins: institution.userInstitutions,
      pendingRequests: pendingByInstitution[institution.id] || [],
    })),
  });
});

/**
 * PATCH /api/admin/institutions
 * Update institution settings (admin only)
 */
export const PATCH = withErrorHandler(async (request: NextRequest) => {
  await getAdminSessionOrRedirect();

  const bodySchema = z.object({
    institutionId: z.string(),
    requiresApproval: z.boolean(),
  });

  const body = await request.json();
  const { institutionId, requiresApproval } = bodySchema.parse(body);

  const institution = await prisma.institution.update({
    where: { id: institutionId },
    data: { requiresApproval },
    select: {
      id: true,
      name: true,
      requiresApproval: true,
    },
  });

  return NextResponse.json({
    success: true,
    message: `Approval requirement ${requiresApproval ? "enabled" : "disabled"} for ${institution.name}`,
    institution,
  });
});
