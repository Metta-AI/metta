import "server-only";

import { InstitutionRepository } from "../data/institution-repository";
import { NotFoundError, ConflictError, AuthorizationError } from "@/lib/errors";

/**
 * Institution Membership Service
 *
 * Business logic layer for institution membership operations.
 * Handles membership workflows, approval logic, and domain-based auto-approval.
 */

export interface JoinInstitutionResult {
  membershipId: string;
  status: "PENDING" | "APPROVED";
  message: string;
  requiresApproval: boolean;
}

export interface MembershipApprovalResult {
  userId: string;
  status: "APPROVED" | "REJECTED";
  message: string;
}

export class InstitutionMembershipService {
  /**
   * Join an institution
   *
   * Handles:
   * - Domain-based auto-approval
   * - Manual approval requirements
   * - Duplicate membership checks
   */
  static async joinInstitution(
    userId: string,
    institutionId: string,
    userEmail: string | null,
    options?: {
      role?: string;
      department?: string;
      title?: string;
    }
  ): Promise<JoinInstitutionResult> {
    // Get institution details
    const institution =
      await InstitutionRepository.findByIdWithBasicInfo(institutionId);

    if (!institution) {
      throw new NotFoundError("Institution", institutionId);
    }

    // Check for existing membership
    const existingMembership = await InstitutionRepository.findMembership(
      userId,
      institutionId
    );

    if (existingMembership) {
      if (existingMembership.status === "PENDING") {
        throw new ConflictError(
          "Your request to join this institution is already pending approval"
        );
      } else if (existingMembership.status === "REJECTED") {
        throw new ConflictError(
          "Your request to join this institution was previously rejected"
        );
      } else {
        throw new ConflictError("You are already a member of this institution");
      }
    }

    // Check for domain-based auto-approval
    const domainAutoApproved = this.checkDomainAutoApproval(
      userEmail,
      institution.domain
    );

    // Determine membership status
    const requiresApproval =
      institution.requiresApproval && !domainAutoApproved;
    const status = requiresApproval ? "PENDING" : "APPROVED";
    const isActive = !requiresApproval;

    // Create membership
    const membership = await InstitutionRepository.createMembership({
      userId,
      institutionId,
      role: options?.role,
      department: options?.department,
      title: options?.title,
      status,
      isActive,
    });

    return {
      membershipId: membership.userId + membership.institutionId,
      status,
      message: requiresApproval
        ? "Your request has been submitted and is pending approval"
        : "You have successfully joined the institution",
      requiresApproval,
    };
  }

  /**
   * Approve or reject a membership request
   *
   * Only admins can approve/reject requests
   */
  static async approveMembership(
    adminUserId: string,
    targetUserId: string,
    institutionId: string,
    approve: boolean
  ): Promise<MembershipApprovalResult> {
    // Check if requester is an admin
    const isAdmin = await InstitutionRepository.isAdmin(
      adminUserId,
      institutionId
    );

    if (!isAdmin) {
      throw new AuthorizationError(
        "You must be an admin to approve or reject membership requests"
      );
    }

    // Get the pending membership
    const membership = await InstitutionRepository.findMembership(
      targetUserId,
      institutionId
    );

    if (!membership) {
      throw new NotFoundError("Membership request");
    }

    if (membership.status !== "PENDING") {
      throw new ConflictError(
        "This membership request has already been processed"
      );
    }

    // Update membership status
    const newStatus = approve ? "APPROVED" : "REJECTED";
    await InstitutionRepository.updateMembership(targetUserId, institutionId, {
      status: newStatus,
      isActive: approve,
    });

    return {
      userId: targetUserId,
      status: newStatus,
      message: approve
        ? "Membership request approved"
        : "Membership request rejected",
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
    institutionId: string,
    newRole: string
  ): Promise<void> {
    // Check if requester is an admin
    const isAdmin = await InstitutionRepository.isAdmin(
      adminUserId,
      institutionId
    );

    if (!isAdmin) {
      throw new AuthorizationError(
        "You must be an admin to change member roles"
      );
    }

    // Get the membership
    const membership = await InstitutionRepository.findMembership(
      targetUserId,
      institutionId
    );

    if (!membership) {
      throw new NotFoundError("Member", targetUserId);
    }

    // Don't allow demoting the last owner
    if (membership.role === "owner" && newRole !== "owner") {
      const adminCount = await InstitutionRepository.countAdmins(institutionId);
      if (adminCount <= 1) {
        throw new ConflictError(
          "Cannot demote the last owner of the institution"
        );
      }
    }

    // Update the role
    await InstitutionRepository.updateMembership(targetUserId, institutionId, {
      role: newRole,
    });
  }

  /**
   * Remove a member from an institution
   *
   * Admins can remove members, members can remove themselves
   */
  static async removeMember(
    requesterId: string,
    targetUserId: string,
    institutionId: string
  ): Promise<void> {
    // Check if requester is admin or removing themselves
    const isAdmin = await InstitutionRepository.isAdmin(
      requesterId,
      institutionId
    );
    const isSelf = requesterId === targetUserId;

    if (!isAdmin && !isSelf) {
      throw new AuthorizationError(
        "You must be an admin to remove other members"
      );
    }

    // Get the target membership
    const membership = await InstitutionRepository.findMembership(
      targetUserId,
      institutionId
    );

    if (!membership) {
      throw new NotFoundError("Member", targetUserId);
    }

    // Don't allow removing the last owner
    if (membership.role === "owner") {
      const adminCount = await InstitutionRepository.countAdmins(institutionId);
      if (adminCount <= 1) {
        throw new ConflictError(
          "Cannot remove the last owner of the institution"
        );
      }
    }

    // Delete the membership
    await InstitutionRepository.deleteMembership(targetUserId, institutionId);
  }

  /**
   * Transfer ownership to another member
   *
   * Only owners can transfer ownership
   */
  static async transferOwnership(
    currentOwnerId: string,
    newOwnerId: string,
    institutionId: string
  ): Promise<void> {
    // Check if requester is an owner
    const isOwner = await InstitutionRepository.isOwner(
      currentOwnerId,
      institutionId
    );

    if (!isOwner) {
      throw new AuthorizationError(
        "You must be an owner to transfer ownership"
      );
    }

    // Get the target membership
    const newOwnerMembership = await InstitutionRepository.findMembership(
      newOwnerId,
      institutionId
    );

    if (!newOwnerMembership || !newOwnerMembership.isActive) {
      throw new NotFoundError("Target member");
    }

    // Transfer ownership (current owner becomes admin, new owner becomes owner)
    await InstitutionRepository.updateMembership(
      currentOwnerId,
      institutionId,
      { role: "admin" }
    );

    await InstitutionRepository.updateMembership(newOwnerId, institutionId, {
      role: "owner",
    });
  }

  /**
   * Check if a user's email domain matches institution domain for auto-approval
   */
  private static checkDomainAutoApproval(
    userEmail: string | null,
    institutionDomain: string | null
  ): boolean {
    if (!userEmail || !institutionDomain) {
      return false;
    }

    const userDomain = userEmail.split("@")[1];
    if (!userDomain) {
      return false;
    }

    return userDomain.toLowerCase() === institutionDomain.toLowerCase();
  }
}
