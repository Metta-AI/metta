"use server";

import { revalidatePath } from "next/cache";
import { zfd } from "zod-form-data";
import { z } from "zod/v4";

import { actionClient } from "@/lib/actionClient";
import { getSessionOrRedirect } from "@/lib/auth";
import { validateInstitutionName } from "@/lib/name-validation";
import { AuthorizationError, ConflictError } from "@/lib/errors";
import { InstitutionRepository } from "../data/institution-repository";

const inputSchema = zfd.formData({
  institutionId: zfd.text(z.string()),
  name: zfd.text(z.string().min(1).max(255)),
  domain: zfd.text(z.string().optional()),
  description: zfd.text(z.string().optional()),
  website: zfd.text(z.string().url().optional().or(z.literal(""))),
  location: zfd.text(z.string().optional()),
  type: zfd.text(
    z.enum([
      "UNIVERSITY",
      "COMPANY",
      "RESEARCH_LAB",
      "NONPROFIT",
      "GOVERNMENT",
      "OTHER",
    ])
  ),
});

export const updateInstitutionAction = actionClient
  .inputSchema(inputSchema)
  .action(async ({ parsedInput: input }) => {
    const session = await getSessionOrRedirect();

    // Check if user has admin or owner rights for this institution
    const isAdmin = await InstitutionRepository.isAdmin(
      session.user.id,
      input.institutionId
    );

    if (!isAdmin) {
      throw new AuthorizationError(
        "You don't have permission to edit this institution"
      );
    }

    // Validate name uniqueness across all entity types (excluding current institution)
    if (input.name) {
      await validateInstitutionName(input.name, input.institutionId);
    }

    // Check domain uniqueness if provided (excluding current institution)
    if (input.domain) {
      const existingDomain = await InstitutionRepository.findByDomain(
        input.domain
      );

      if (existingDomain && existingDomain.id !== input.institutionId) {
        throw new ConflictError(
          "An institution with this domain already exists"
        );
      }
    }

    // Update the institution
    const institution = await InstitutionRepository.update(
      input.institutionId,
      {
        name: input.name,
        domain: input.domain || null,
        description: input.description || null,
        website: input.website || null,
        location: input.location || null,
        type: input.type,
      }
    );

    revalidatePath("/institutions");

    return {
      id: institution.id,
      name: institution.name,
      message: "Institution updated successfully",
    };
  });
