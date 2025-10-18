"use server";

import { revalidatePath } from "next/cache";
import { zfd } from "zod-form-data";
import { z } from "zod/v4";

import { actionClient } from "@/lib/actionClient";
import { getSessionOrRedirect } from "@/lib/auth";
import { prisma } from "@/lib/db/prisma";
import { InstitutionMembershipService } from "../services/membership-service";
import { InstitutionRepository } from "../data/institution-repository";
import { NotFoundError } from "@/lib/errors";

const inputSchema = zfd.formData({
  institutionId: zfd.text(z.string()),
  userEmail: zfd.text(z.string().email()),
  role: zfd.text(z.string().optional()),
  department: zfd.text(z.string().optional()),
  title: zfd.text(z.string().optional()),
  action: zfd.text(z.enum(["add", "update", "remove"])),
});

export const manageUserMembershipAction = actionClient
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
      case "add":
        await InstitutionRepository.createMembership({
          userId: targetUser.id,
          institutionId: input.institutionId,
          role: input.role || "member",
          department: input.department,
          title: input.title,
          isActive: true,
        });
        break;

      case "update":
        if (input.role) {
          await InstitutionMembershipService.updateMemberRole(
            session.user.id,
            targetUser.id,
            input.institutionId,
            input.role
          );
        }
        // Update other fields if provided
        if (input.department || input.title) {
          await InstitutionRepository.updateMembership(
            targetUser.id,
            input.institutionId,
            {
              department: input.department || null,
              title: input.title || null,
            }
          );
        }
        break;

      case "remove":
        await InstitutionMembershipService.removeMember(
          session.user.id,
          targetUser.id,
          input.institutionId
        );
        break;
    }

    revalidatePath("/institutions");

    return {
      success: true,
      message: `User ${input.action === "add" ? "added to" : input.action === "update" ? "updated in" : "removed from"} institution successfully`,
    };
  });
