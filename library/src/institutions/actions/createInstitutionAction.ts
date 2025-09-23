"use server";

import { revalidatePath } from "next/cache";
import { zfd } from "zod-form-data";
import { z } from "zod/v4";

import { actionClient } from "@/lib/actionClient";
import { getSessionOrRedirect } from "@/lib/auth";
import { prisma } from "@/lib/db/prisma";

const inputSchema = zfd.formData({
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

export const createInstitutionAction = actionClient
  .inputSchema(inputSchema)
  .action(async ({ parsedInput: input }) => {
    const session = await getSessionOrRedirect();

    // Check if institution with this name already exists
    const existingInstitution = await prisma.institution.findUnique({
      where: { name: input.name },
    });

    if (existingInstitution) {
      throw new Error("An institution with this name already exists");
    }

    // Check domain uniqueness if provided
    if (input.domain) {
      const existingDomain = await prisma.institution.findUnique({
        where: { domain: input.domain },
      });

      if (existingDomain) {
        throw new Error("An institution with this domain already exists");
      }
    }

    // Create the institution
    const institution = await prisma.institution.create({
      data: {
        name: input.name,
        domain: input.domain || null,
        description: input.description || null,
        website: input.website || null,
        location: input.location || null,
        type: input.type,
        createdByUserId: session.user.id,
      },
      select: {
        id: true,
        name: true,
      },
    });

    // Automatically add the creator as an admin of the institution
    await prisma.userInstitution.create({
      data: {
        userId: session.user.id,
        institutionId: institution.id,
        role: "admin",
        isActive: true,
      },
    });

    revalidatePath("/institutions");

    return {
      id: institution.id,
      name: institution.name,
      message: "Institution created successfully",
    };
  });

