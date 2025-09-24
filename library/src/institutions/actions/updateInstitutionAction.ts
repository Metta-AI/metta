"use server";

import { revalidatePath } from "next/cache";
import { zfd } from "zod-form-data";
import { z } from "zod/v4";

import { actionClient } from "@/lib/actionClient";
import { getSessionOrRedirect } from "@/lib/auth";
import { prisma } from "@/lib/db/prisma";

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
    const userInstitution = await prisma.userInstitution.findUnique({
      where: {
        userId_institutionId: {
          userId: session.user.id,
          institutionId: input.institutionId,
        },
      },
    });

    if (!userInstitution || userInstitution.role !== "admin") {
      throw new Error("You don't have permission to edit this institution");
    }

    // Check if institution with this name already exists (excluding current institution)
    if (input.name) {
      const existingInstitution = await prisma.institution.findFirst({
        where: {
          name: input.name,
          id: { not: input.institutionId },
        },
      });

      if (existingInstitution) {
        throw new Error("An institution with this name already exists");
      }
    }

    // Check domain uniqueness if provided (excluding current institution)
    if (input.domain) {
      const existingDomain = await prisma.institution.findFirst({
        where: {
          domain: input.domain,
          id: { not: input.institutionId },
        },
      });

      if (existingDomain) {
        throw new Error("An institution with this domain already exists");
      }
    }

    // Update the institution
    const institution = await prisma.institution.update({
      where: { id: input.institutionId },
      data: {
        name: input.name,
        domain: input.domain || null,
        description: input.description || null,
        website: input.website || null,
        location: input.location || null,
        type: input.type,
      },
      select: {
        id: true,
        name: true,
      },
    });

    revalidatePath("/institutions");

    return {
      id: institution.id,
      name: institution.name,
      message: "Institution updated successfully",
    };
  });
