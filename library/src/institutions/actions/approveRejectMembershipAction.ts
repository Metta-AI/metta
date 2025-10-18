"use server";

import { revalidatePath } from "next/cache";
import { zfd } from "zod-form-data";
import { z } from "zod/v4";

import { actionClient } from "@/lib/actionClient";
import { getSessionOrRedirect } from "@/lib/auth";
import { prisma } from "@/lib/db/prisma";

const inputSchema = zfd.formData({
  institutionId: zfd.text(z.string()),
  userEmail: zfd.text(z.string().email()),
  action: zfd.text(z.enum(["approve", "reject"])),
});

export const approveRejectMembershipAction = actionClient
  .inputSchema(inputSchema)
  .action(async ({ parsedInput: input }) => {
    const session = await getSessionOrRedirect();

    // Check if user has admin rights for this institution
    const userInstitution = await prisma.userInstitution.findUnique({
      where: {
        userId_institutionId: {
          userId: session.user.id,
          institutionId: input.institutionId,
        },
      },
    });

    if (!userInstitution || userInstitution.role !== "admin") {
      throw new Error(
        "You don't have permission to manage membership requests for this institution"
      );
    }

    // Find the target user by email
    const targetUser = await prisma.user.findUnique({
      where: { email: input.userEmail },
    });

    if (!targetUser) {
      throw new Error("User not found");
    }

    // Find the pending membership
    const pendingMembership = await prisma.userInstitution.findUnique({
      where: {
        userId_institutionId: {
          userId: targetUser.id,
          institutionId: input.institutionId,
        },
      },
      include: {
        institution: {
          select: { name: true },
        },
      },
    });

    if (!pendingMembership) {
      throw new Error("No membership request found for this user");
    }

    if (pendingMembership.status !== "PENDING") {
      throw new Error("This membership request is not pending approval");
    }

    // Update the membership status
    const newStatus = input.action === "approve" ? "APPROVED" : "REJECTED";
    const isActive = input.action === "approve";

    await prisma.userInstitution.update({
      where: {
        userId_institutionId: {
          userId: targetUser.id,
          institutionId: input.institutionId,
        },
      },
      data: {
        status: newStatus,
        isActive: isActive,
      },
    });

    revalidatePath("/institutions");

    return {
      success: true,
      message: `Membership request ${input.action === "approve" ? "approved" : "rejected"} for ${targetUser.name || targetUser.email} at ${pendingMembership.institution.name}`,
      action: input.action,
    };
  });
