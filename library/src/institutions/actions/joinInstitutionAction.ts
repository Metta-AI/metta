"use server";

import { revalidatePath } from "next/cache";
import { zfd } from "zod-form-data";
import { z } from "zod/v4";

import { actionClient } from "@/lib/actionClient";
import { getSessionOrRedirect } from "@/lib/auth";
import { InstitutionMembershipService } from "../services/membership-service";
import { InstitutionRepository } from "../data/institution-repository";

const inputSchema = zfd.formData({
  institutionId: zfd.text(z.string()),
  role: zfd.text(z.string().optional()),
  department: zfd.text(z.string().optional()),
  title: zfd.text(z.string().optional()),
});

export const joinInstitutionAction = actionClient
  .inputSchema(inputSchema)
  .action(async ({ parsedInput: input }) => {
    const session = await getSessionOrRedirect();

    // Use service to handle join logic
    const result = await InstitutionMembershipService.joinInstitution(
      session.user.id,
      input.institutionId,
      session.user.email,
      {
        role: input.role,
        department: input.department,
        title: input.title,
      }
    );

    // Get institution name for response message
    const institution = await InstitutionRepository.findByIdWithBasicInfo(
      input.institutionId
    );

    revalidatePath("/institutions");

    return {
      success: true,
      status: result.status.toLowerCase(),
      message: result.requiresApproval
        ? `Your request to join ${institution?.name} has been submitted and is pending approval from an administrator`
        : `Successfully joined ${institution?.name}`,
      institutionName: institution?.name,
    };
  });
