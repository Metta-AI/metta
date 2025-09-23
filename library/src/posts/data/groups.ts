import { prisma } from "@/lib/db/prisma";
import { auth } from "@/lib/auth";

export type GroupDTO = {
  id: string;
  name: string;
  description: string | null;
  isPublic: boolean;
  createdAt: Date;
  memberCount: number;
  currentUserRole: string | null;
  members?: Array<{
    id: string;
    user: {
      name: string | null;
      email: string | null;
    };
    role: string | null;
    joinedAt: Date;
    isActive: boolean;
  }>;
};

/**
 * Load groups where current user is a member (with full details including members)
 */
export async function loadUserGroups(): Promise<GroupDTO[]> {
  const session = await auth();

  if (!session?.user?.id) {
    return [];
  }

  const groups = await prisma.group.findMany({
    where: {
      userGroups: {
        some: {
          userId: session.user.id,
          isActive: true,
        },
      },
    },
    include: {
      userGroups: {
        where: { isActive: true },
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
    },
    orderBy: { createdAt: "desc" },
  });

  return groups.map((group) => {
    const currentUserMembership = group.userGroups.find(
      (ug) => ug.userId === session.user.id
    );

    return {
      id: group.id,
      name: group.name,
      description: group.description,
      isPublic: group.isPublic,
      createdAt: group.createdAt,
      memberCount: group.userGroups.length,
      currentUserRole: currentUserMembership?.role || null,
      members: group.userGroups.map((ug) => ({
        id: ug.id,
        user: ug.user,
        role: ug.role,
        joinedAt: ug.joinedAt,
        isActive: ug.isActive,
      })),
    };
  });
}

/**
 * Load all public groups (for discovery)
 */
export async function loadAllPublicGroups(): Promise<GroupDTO[]> {
  const session = await auth();

  const groups = await prisma.group.findMany({
    where: {
      isPublic: true,
    },
    include: {
      userGroups: {
        where: {
          isActive: true,
        },
      },
    },
    orderBy: [{ createdAt: "desc" }],
  });

  return groups.map((group) => {
    const currentUserMembership = session?.user?.id
      ? group.userGroups.find((ug) => ug.userId === session.user.id)
      : null;

    const currentUserRole = currentUserMembership?.role || null;

    return {
      id: group.id,
      name: group.name,
      description: group.description,
      isPublic: group.isPublic,
      createdAt: group.createdAt,
      memberCount: group.userGroups.length,
      currentUserRole,
    };
  });
}

/**
 * Load a specific group with full details (for management)
 */
export async function loadGroupById(groupId: string): Promise<GroupDTO | null> {
  const session = await auth();

  if (!session?.user?.id) {
    return null;
  }

  const group = await prisma.group.findUnique({
    where: { id: groupId },
    include: {
      userGroups: {
        where: { isActive: true },
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
    },
  });

  if (!group) {
    return null;
  }

  // Check if user is a member or if group is public
  const currentUserMembership = group.userGroups.find(
    (ug) => ug.userId === session.user.id
  );

  if (!currentUserMembership && !group.isPublic) {
    return null; // Private group and user is not a member
  }

  return {
    id: group.id,
    name: group.name,
    description: group.description,
    isPublic: group.isPublic,
    createdAt: group.createdAt,
    memberCount: group.userGroups.length,
    currentUserRole: currentUserMembership?.role || null,
    members: currentUserMembership
      ? group.userGroups.map((ug) => ({
          id: ug.id,
          user: ug.user,
          role: ug.role,
          joinedAt: ug.joinedAt,
          isActive: ug.isActive,
        }))
      : undefined, // Only include members if user is a member
  };
}
