import "server-only";

import { prisma } from "@/lib/db/prisma";
import type {
  Institution,
  UserInstitution,
  InstitutionType,
  UserInstitutionStatus,
} from "@prisma/client";

/**
 * Institution Repository
 *
 * Data access layer for institution-related database operations.
 * All Prisma queries for institutions should go through this repository.
 */

export interface InstitutionWithMembershipCount extends Institution {
  _count?: {
    members: number;
  };
}

export interface UserInstitutionWithDetails extends UserInstitution {
  user?: {
    id: string;
    name: string | null;
    email: string | null;
  };
  institution?: {
    id: string;
    name: string;
  };
}

export class InstitutionRepository {
  /**
   * Find an institution by ID
   */
  static async findById(institutionId: string): Promise<Institution | null> {
    return prisma.institution.findUnique({
      where: { id: institutionId },
    });
  }

  /**
   * Find an institution by ID with basic info
   */
  static async findByIdWithBasicInfo(institutionId: string) {
    return prisma.institution.findUnique({
      where: { id: institutionId },
      select: {
        id: true,
        name: true,
        domain: true,
        requiresApproval: true,
      },
    });
  }

  /**
   * Find an institution by domain
   */
  static async findByDomain(domain: string): Promise<Institution | null> {
    return prisma.institution.findUnique({
      where: { domain },
    });
  }

  /**
   * Find a user's membership in an institution
   */
  static async findMembership(
    userId: string,
    institutionId: string
  ): Promise<UserInstitution | null> {
    return prisma.userInstitution.findUnique({
      where: {
        userId_institutionId: {
          userId,
          institutionId,
        },
      },
    });
  }

  /**
   * Find all memberships for an institution with user details
   */
  static async findMembershipsWithUsers(institutionId: string) {
    return prisma.userInstitution.findMany({
      where: { institutionId },
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
   * Find pending membership requests for an institution
   */
  static async findPendingMemberships(institutionId: string) {
    return prisma.userInstitution.findMany({
      where: {
        institutionId,
        status: "PENDING",
      },
      include: {
        user: {
          select: {
            id: true,
            name: true,
            email: true,
          },
        },
      },
      orderBy: { userId: "asc" },
    });
  }

  /**
   * Find all institutions a user belongs to
   */
  static async findUserInstitutions(userId: string) {
    return prisma.userInstitution.findMany({
      where: {
        userId,
        isActive: true,
      },
      include: {
        institution: true,
      },
    });
  }

  /**
   * Create a new institution
   */
  static async create(data: {
    name: string;
    domain: string | null;
    description: string | null;
    website: string | null;
    location: string | null;
    type: InstitutionType;
    createdByUserId: string;
  }) {
    return prisma.institution.create({
      data,
      select: {
        id: true,
        name: true,
      },
    });
  }

  /**
   * Update an institution
   */
  static async update(
    institutionId: string,
    data: {
      name?: string;
      domain?: string | null;
      description?: string | null;
      website?: string | null;
      location?: string | null;
      type?: InstitutionType;
      requiresApproval?: boolean;
    }
  ) {
    return prisma.institution.update({
      where: { id: institutionId },
      data,
    });
  }

  /**
   * Create a membership
   */
  static async createMembership(data: {
    userId: string;
    institutionId: string;
    role?: string;
    department?: string | null;
    title?: string | null;
    status?: UserInstitutionStatus;
    isActive?: boolean;
  }) {
    return prisma.userInstitution.create({
      data: {
        userId: data.userId,
        institutionId: data.institutionId,
        role: data.role || "member",
        department: data.department || null,
        title: data.title || null,
        status: data.status || "APPROVED",
        isActive: data.isActive ?? true,
      },
    });
  }

  /**
   * Update a membership
   */
  static async updateMembership(
    userId: string,
    institutionId: string,
    data: {
      role?: string;
      status?: UserInstitutionStatus;
      isActive?: boolean;
      department?: string | null;
      title?: string | null;
    }
  ) {
    return prisma.userInstitution.update({
      where: {
        userId_institutionId: {
          userId,
          institutionId,
        },
      },
      data,
    });
  }

  /**
   * Delete a membership
   */
  static async deleteMembership(userId: string, institutionId: string) {
    return prisma.userInstitution.delete({
      where: {
        userId_institutionId: {
          userId,
          institutionId,
        },
      },
    });
  }

  /**
   * Check if a user is an admin of an institution
   */
  static async isAdmin(
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
        role: true,
        isActive: true,
      },
    });

    return membership?.role === "admin" && membership?.isActive === true;
  }

  /**
   * Check if a user is an owner of an institution
   */
  static async isOwner(
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
        role: true,
        isActive: true,
      },
    });

    return membership?.role === "owner" && membership?.isActive === true;
  }

  /**
   * Count admins for an institution
   */
  static async countAdmins(institutionId: string): Promise<number> {
    return prisma.userInstitution.count({
      where: {
        institutionId,
        role: { in: ["admin", "owner"] },
        isActive: true,
      },
    });
  }
}
