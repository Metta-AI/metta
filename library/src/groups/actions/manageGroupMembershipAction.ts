"use server";

import { revalidatePath } from "next/cache";
import { zfd } from "zod-form-data";
import { z } from "zod/v4";

import { actionClient } from "@/lib/actionClient";
import { getSessionOrRedirect } from "@/lib/auth";
import { prisma } from "@/lib/db/prisma";
import { GroupMembershipService } from "../services/membership-service";
import { GroupRepository } from "../data/group-repository";
import { NotFoundError, ConflictError, AuthorizationError } from "@/lib/errors";

const inputSchema = zfd.formData({
  groupId: zfd.text(z.string()),
  userEmail: zfd.text(z.string().email()),
  role: zfd.text(z.string().optional()),
  action: zfd.text(z.enum(["add", "update", "remove"])),
});

export const manageGroupMembershipAction = actionClient
  .inputSchema(inputSchema)
  .action(async ({ parsedInput: input }) => {
    const session = await getSessionOrRedirect();

    // Find the target user by email
    const targetUser = await prisma.user.findUnique({
      where: { email: input.userEmail },
      select: { id: true, name: true, email: true },
    });

    if (!targetUser) {
      throw new NotFoundError("User", input.userEmail);
    }

    // Use service layer for different actions
    switch (input.action) {
      case "add": {
        // Get the group to check institution
        const group = await GroupRepository.findByIdWithBasicInfo(
          input.groupId
        );
        if (!group) {
          throw new NotFoundError("Group", input.groupId);
        }

        // Check if target user is a member of the same institution
        const isInInstitution = await GroupRepository.isUserInInstitution(
          targetUser.id,
          group.institutionId
        );

        if (!isInInstitution) {
          throw new AuthorizationError(
            "User must be a member of the same institution to join this group"
          );
        }

        // Check for existing membership
        const existingMembership = await GroupRepository.findMembership(
          targetUser.id,
          input.groupId
        );

        if (existingMembership) {
          throw new ConflictError("User is already a member of this group");
        }

        await GroupRepository.createMembership({
          userId: targetUser.id,
          groupId: input.groupId,
          role: input.role || "member",
          isActive: true,
        });
        break;
      }

      case "update":
        if (input.role) {
          await GroupMembershipService.updateMemberRole(
            session.user.id,
            targetUser.id,
            input.groupId,
            input.role
          );
        }
        break;

      case "remove":
        await GroupMembershipService.removeMember(
          session.user.id,
          targetUser.id,
          input.groupId
        );
        break;
    }

    revalidatePath("/groups");

    return {
      success: true,
      message: `User ${input.action === "add" ? "added to" : input.action === "update" ? "updated in" : "removed from"} group successfully`,
    };
  });
