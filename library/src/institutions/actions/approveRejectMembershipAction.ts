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
  action: zfd.text(z.enum(["approve", "reject"])),
});

export const approveRejectMembershipAction = actionClient
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

    // Use service to handle approval/rejection
    const result = await InstitutionMembershipService.approveMembership(
      session.user.id,
      targetUser.id,
      input.institutionId,
      input.action === "approve"
    );

    // Get institution name for response message
    const institution = await InstitutionRepository.findByIdWithBasicInfo(
      input.institutionId
    );

    revalidatePath("/institutions");

    return {
      success: true,
      message: `Membership request ${input.action === "approve" ? "approved" : "rejected"} for ${targetUser.name || targetUser.email} at ${institution?.name}`,
      action: input.action,
    };
  });
