import "server-only";

import { GroupRepository } from "../data/group-repository";
import { NotFoundError, ConflictError, AuthorizationError } from "@/lib/errors";

/**
 * Group Membership Service
 *
 * Business logic layer for group membership operations.
 * Handles membership workflows and permission checks.
 */

export interface JoinGroupResult {
  membershipId: string;
  message: string;
}

export class GroupMembershipService {
  /**
   * Join a public group
   *
   * Handles:
   * - Institution membership verification
   * - Public/private group checks
   * - Duplicate membership checks
   */
  static async joinGroup(
    userId: string,
    groupId: string
  ): Promise<JoinGroupResult> {
    // Get group details
    const group = await GroupRepository.findByIdWithBasicInfo(groupId);

    if (!group) {
      throw new NotFoundError("Group", groupId);
    }

    // Check if group is public
    if (!group.isPublic) {
      throw new AuthorizationError(
        "This group is private and requires an invitation"
      );
    }

    // Check if user is a member of the same institution
    const isInInstitution = await GroupRepository.isUserInInstitution(
      userId,
      group.institutionId
    );

    if (!isInInstitution) {
      throw new AuthorizationError(
        "You must be a member of the same institution to join this group"
      );
    }

    // Check for existing membership
    const existingMembership = await GroupRepository.findMembership(
      userId,
      groupId
    );

    if (existingMembership) {
      throw new ConflictError("You are already a member of this group");
    }

    // Create membership
    const membership = await GroupRepository.createMembership({
      userId,
      groupId,
      role: "member",
      isActive: true,
    });

    return {
      membershipId: membership.userId + membership.groupId,
      message: "Successfully joined the group",
    };
  }

  /**
   * Update a member's role
   *
   * Only admins can change roles
   */
  static async updateMemberRole(
    adminUserId: string,
    targetUserId: string,
    groupId: string,
    newRole: string
  ): Promise<void> {
    // Check if requester is an admin
    const isAdmin = await GroupRepository.isAdmin(adminUserId, groupId);

    if (!isAdmin) {
      throw new AuthorizationError(
        "You must be an admin to change member roles"
      );
    }

    // Get the membership
    const membership = await GroupRepository.findMembership(
      targetUserId,
      groupId
    );

    if (!membership) {
      throw new NotFoundError("Member", targetUserId);
    }

    // Don't allow demoting the last admin
    if (membership.role === "admin" && newRole !== "admin") {
      const adminCount = await GroupRepository.countAdmins(groupId);
      if (adminCount <= 1) {
        throw new ConflictError("Cannot demote the last admin of the group");
      }
    }

    // Update the role
    await GroupRepository.updateMembership(targetUserId, groupId, {
      role: newRole,
    });
  }

  /**
   * Remove a member from a group
   *
   * Admins can remove members, members can remove themselves
   */
  static async removeMember(
    requesterId: string,
    targetUserId: string,
    groupId: string
  ): Promise<void> {
    // Check if requester is admin or removing themselves
    const isAdmin = await GroupRepository.isAdmin(requesterId, groupId);
    const isSelf = requesterId === targetUserId;

    if (!isAdmin && !isSelf) {
      throw new AuthorizationError(
        "You must be an admin to remove other members"
      );
    }

    // Get the target membership
    const membership = await GroupRepository.findMembership(
      targetUserId,
      groupId
    );

    if (!membership) {
      throw new NotFoundError("Member", targetUserId);
    }

    // Don't allow removing the last admin
    if (membership.role === "admin") {
      const adminCount = await GroupRepository.countAdmins(groupId);
      if (adminCount <= 1) {
        throw new ConflictError("Cannot remove the last admin of the group");
      }
    }

    // Delete the membership
    await GroupRepository.deleteMembership(targetUserId, groupId);
  }
}
