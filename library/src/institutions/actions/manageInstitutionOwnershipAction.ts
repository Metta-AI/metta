"use server";

import { revalidatePath } from "next/cache";
import { zfd } from "zod-form-data";
import { z } from "zod/v4";

import { actionClient } from "@/lib/actionClient";
import { getAdminSessionOrRedirect } from "@/lib/adminAuth";
import { prisma } from "@/lib/db/prisma";
import { InstitutionRepository } from "../data/institution-repository";
import { NotFoundError, ConflictError } from "@/lib/errors";

const inputSchema = zfd.formData({
  institutionId: zfd.text(z.string()),
  userEmail: zfd.text(z.string().email()),
  action: zfd.text(z.enum(["assign_admin", "remove_admin"])),
});

export const manageInstitutionOwnershipAction = actionClient
  .inputSchema(inputSchema)
  .action(async ({ parsedInput: input }) => {
    // Verify global admin access
    await getAdminSessionOrRedirect();

    // Find the institution
    const institution = await InstitutionRepository.findById(
      input.institutionId
    );

    if (!institution) {
      throw new NotFoundError("Institution", input.institutionId);
    }

    // Find the target user by email
    const targetUser = await prisma.user.findUnique({
      where: { email: input.userEmail },
      select: { id: true, name: true, email: true },
    });

    if (!targetUser) {
      throw new NotFoundError("User", input.userEmail);
    }

    switch (input.action) {
      case "assign_admin": {
        // Check if user is already a member
        const existingMembership = await InstitutionRepository.findMembership(
          targetUser.id,
          input.institutionId
        );

        if (existingMembership) {
          // Update existing membership to admin
          await InstitutionRepository.updateMembership(
            targetUser.id,
            input.institutionId,
            {
              role: "admin",
              isActive: true,
            }
          );
        } else {
          // Create new admin membership
          await InstitutionRepository.createMembership({
            userId: targetUser.id,
            institutionId: input.institutionId,
            role: "admin",
            isActive: true,
          });
        }
        break;
      }

      case "remove_admin": {
        // Check if this is the last admin
        const adminCount = await InstitutionRepository.countAdmins(
          input.institutionId
        );
        const targetMembership = await InstitutionRepository.findMembership(
          targetUser.id,
          input.institutionId
        );

        if (targetMembership?.role === "admin" && adminCount <= 1) {
          throw new ConflictError(
            "Cannot remove the last admin of the institution"
          );
        }

        // Convert to member
        if (targetMembership?.role === "admin") {
          await InstitutionRepository.updateMembership(
            targetUser.id,
            input.institutionId,
            {
              role: "member",
            }
          );
        }
        break;
      }
    }

    revalidatePath("/admin/institutions");

    return {
      success: true,
      message: `Successfully ${input.action.replace("_", " ")} for ${institution.name}`,
    };
  });
