import "server-only";

import { prisma } from "@/lib/db/prisma";
import type { Group, UserGroup } from "@prisma/client";

/**
 * Group Repository
 *
 * Data access layer for group-related database operations.
 * All Prisma queries for groups should go through this repository.
 */

export interface GroupWithMembershipCount extends Group {
  _count?: {
    members: number;
  };
}

export interface UserGroupWithDetails extends UserGroup {
  user?: {
    id: string;
    name: string | null;
    email: string | null;
  };
  group?: {
    id: string;
    name: string;
  };
}

export class GroupRepository {
  /**
   * Find a group by ID
   */
  static async findById(groupId: string): Promise<Group | null> {
    return prisma.group.findUnique({
      where: { id: groupId },
    });
  }

  /**
   * Find a group by ID with basic info
   */
  static async findByIdWithBasicInfo(groupId: string) {
    return prisma.group.findUnique({
      where: { id: groupId },
      select: {
        id: true,
        name: true,
        isPublic: true,
        institutionId: true,
      },
    });
  }

  /**
   * Find a user's membership in a group
   */
  static async findMembership(
    userId: string,
    groupId: string
  ): Promise<UserGroup | null> {
    return prisma.userGroup.findUnique({
      where: {
        userId_groupId: {
          userId,
          groupId,
        },
      },
    });
  }

  /**
   * Find all memberships for a group with user details
   */
  static async findMembershipsWithUsers(groupId: string) {
    return prisma.userGroup.findMany({
      where: { groupId },
      include: {
        user: {
          select: {
            id: true,
            name: true,
            email: true,
          },
        },
      },
      orderBy: { role: "asc" },
    });
  }

  /**
   * Find all groups a user belongs to
   */
  static async findUserGroups(userId: string) {
    return prisma.userGroup.findMany({
      where: {
        userId,
        isActive: true,
      },
      include: {
        group: true,
      },
    });
  }

  /**
   * Create a new group
   */
  static async create(data: {
    name: string;
    description: string | null;
    isPublic: boolean;
    institutionId: string;
    createdByUserId: string;
  }) {
    return prisma.group.create({
      data,
      select: {
        id: true,
        name: true,
      },
    });
  }

  /**
   * Update a group
   */
  static async update(
    groupId: string,
    data: {
      name?: string;
      description?: string | null;
      isPublic?: boolean;
    }
  ) {
    return prisma.group.update({
      where: { id: groupId },
      data,
    });
  }

  /**
   * Delete a group
   */
  static async delete(groupId: string) {
    return prisma.group.delete({
      where: { id: groupId },
    });
  }

  /**
   * Create a membership
   */
  static async createMembership(data: {
    userId: string;
    groupId: string;
    role?: string;
    isActive?: boolean;
  }) {
    return prisma.userGroup.create({
      data: {
        userId: data.userId,
        groupId: data.groupId,
        role: data.role || "member",
        isActive: data.isActive ?? true,
      },
    });
  }

  /**
   * Update a membership
   */
  static async updateMembership(
    userId: string,
    groupId: string,
    data: {
      role?: string;
      isActive?: boolean;
    }
  ) {
    return prisma.userGroup.update({
      where: {
        userId_groupId: {
          userId,
          groupId,
        },
      },
      data,
    });
  }

  /**
   * Delete a membership
   */
  static async deleteMembership(userId: string, groupId: string) {
    return prisma.userGroup.delete({
      where: {
        userId_groupId: {
          userId,
          groupId,
        },
      },
    });
  }

  /**
   * Check if a user is an admin of a group
   */
  static async isAdmin(userId: string, groupId: string): Promise<boolean> {
    const membership = await prisma.userGroup.findUnique({
      where: {
        userId_groupId: {
          userId,
          groupId,
        },
      },
      select: {
        role: true,
        isActive: true,
      },
    });

    return membership?.role === "admin" && membership?.isActive === true;
  }

  /**
   * Count admins for a group
   */
  static async countAdmins(groupId: string): Promise<number> {
    return prisma.userGroup.count({
      where: {
        groupId,
        role: "admin",
        isActive: true,
      },
    });
  }

  /**
   * Check if a user is a member of the group's institution
   */
  static async isUserInInstitution(
    userId: string,
    institutionId: string
  ): Promise<boolean> {
    const membership = await prisma.userInstitution.findUnique({
      where: {
        userId_institutionId: {
          userId,
          institutionId,
        },
      },
      select: {
        isActive: true,
      },
    });

    return membership?.isActive === true;
  }
}
