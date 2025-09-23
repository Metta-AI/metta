import { NextRequest, NextResponse } from "next/server";
import { z } from "zod/v4";

import { getSessionOrRedirect } from "@/lib/auth";
import { prisma } from "@/lib/db/prisma";
import { MentionType } from "@/lib/mentions";

const searchParamsSchema = z.object({
  q: z.string().min(0).max(50), // Query string
  type: z.enum(["user", "group-relative", "group-absolute"]),
  domain: z.string().optional(), // For absolute group searches
  limit: z.coerce.number().min(1).max(20).default(10),
});

interface MentionSuggestion {
  type: MentionType;
  id: string;
  value: string; // What gets inserted (@username, @/groupname, etc.)
  display: string; // What user sees in dropdown
  avatar?: string; // User avatar or group icon
  subtitle?: string; // Additional info (email, member count, etc.)
  memberCount?: number; // For groups
}

export async function GET(request: NextRequest) {
  try {
    const session = await getSessionOrRedirect();

    const { searchParams } = new URL(request.url);
    const params = searchParamsSchema.parse({
      q: searchParams.get("q") || "",
      type: searchParams.get("type"),
      domain: searchParams.get("domain") || undefined,
      limit: searchParams.get("limit"),
    });

    const suggestions: MentionSuggestion[] = [];

    if (params.type === "user") {
      // Search for users
      const users = await prisma.user.findMany({
        where: {
          OR: [
            { name: { contains: params.q, mode: "insensitive" } },
            { email: { contains: params.q, mode: "insensitive" } },
          ],
        },
        select: {
          id: true,
          name: true,
          email: true,
        },
        take: params.limit,
      });

      users.forEach((user) => {
        const username = user.email?.split("@")[0] || user.id;
        suggestions.push({
          type: "user",
          id: user.id,
          value: `@${username}`,
          display: user.name || username,
          subtitle: user.email || undefined,
        });
      });
    } else if (params.type === "group-relative") {
      // Search for groups in user's institutions
      const userInstitutions = await prisma.userInstitution.findMany({
        where: {
          userId: session.user.id,
          isActive: true,
        },
        select: { institutionId: true },
      });

      const institutionIds = userInstitutions.map((ui) => ui.institutionId);

      const groups = await prisma.group.findMany({
        where: {
          name: { contains: params.q, mode: "insensitive" },
          institutionId: { in: institutionIds },
          // Only show public groups or groups user is a member of
          OR: [
            { isPublic: true },
            {
              userGroups: {
                some: {
                  userId: session.user.id,
                  isActive: true,
                },
              },
            },
          ],
        },
        include: {
          institution: {
            select: { name: true },
          },
          _count: {
            select: {
              userGroups: {
                where: { isActive: true },
              },
            },
          },
        },
        take: params.limit,
      });

      groups.forEach((group) => {
        suggestions.push({
          type: "group-relative",
          id: group.id,
          value: `@/${group.name}`,
          display: group.name,
          subtitle: `${group.institution.name} • ${group._count.userGroups} members`,
          memberCount: group._count.userGroups,
        });
      });
    } else if (params.type === "group-absolute") {
      // Search for groups in specific institution by domain
      if (!params.domain) {
        return NextResponse.json({ suggestions: [] });
      }

      const institution = await prisma.institution.findUnique({
        where: { domain: params.domain },
        select: { id: true, name: true },
      });

      if (!institution) {
        return NextResponse.json({ suggestions: [] });
      }

      const groups = await prisma.group.findMany({
        where: {
          name: { contains: params.q, mode: "insensitive" },
          institutionId: institution.id,
          // Only public groups for absolute mentions (unless user is member of institution)
          OR: [
            { isPublic: true },
            {
              // Allow private groups if user is in the same institution
              institution: {
                userInstitutions: {
                  some: {
                    userId: session.user.id,
                    isActive: true,
                  },
                },
              },
            },
          ],
        },
        include: {
          _count: {
            select: {
              userGroups: {
                where: { isActive: true },
              },
            },
          },
        },
        take: params.limit,
      });

      groups.forEach((group) => {
        suggestions.push({
          type: "group-absolute",
          id: group.id,
          value: `@${params.domain}/${group.name}`,
          display: group.name,
          subtitle: `${institution.name} • ${group._count.userGroups} members`,
          memberCount: group._count.userGroups,
        });
      });
    }

    return NextResponse.json({ suggestions });
  } catch (error) {
    console.error("Error searching mentions:", error);
    return NextResponse.json(
      { error: "Failed to search mentions" },
      { status: 500 }
    );
  }
}
