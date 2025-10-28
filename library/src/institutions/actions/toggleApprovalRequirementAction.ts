"use server";

import { revalidatePath } from "next/cache";
import { zfd } from "zod-form-data";
import { z } from "zod/v4";

import { actionClient } from "@/lib/actionClient";
import { getSessionOrRedirect } from "@/lib/auth";
import { prisma } from "@/lib/db/prisma";

const inputSchema = zfd.formData({
  institutionId: zfd.text(z.string()),
  requiresApproval: zfd.text(
    z.enum(["true", "false"]).transform((val) => val === "true")
  ),
});

export const toggleApprovalRequirementAction = actionClient
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
        "You don't have permission to modify settings for this institution"
      );
    }

    // Update the institution's approval requirement
    const institution = await prisma.institution.update({
      where: { id: input.institutionId },
      data: {
        requiresApproval: input.requiresApproval,
      },
      select: {
        id: true,
        name: true,
        requiresApproval: true,
      },
    });

    revalidatePath("/institutions");

    return {
      success: true,
      message: `Approval requirement ${input.requiresApproval ? "enabled" : "disabled"} for ${institution.name}`,
      requiresApproval: institution.requiresApproval,
    };
  });
