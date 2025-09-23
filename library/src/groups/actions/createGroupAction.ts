"use server";

import { revalidatePath } from "next/cache";
import { zfd } from "zod-form-data";
import { z } from "zod/v4";

import { actionClient } from "@/lib/actionClient";
import { getSessionOrRedirect } from "@/lib/auth";
import { prisma } from "@/lib/db/prisma";

const inputSchema = zfd.formData({
  name: zfd.text(
    z
      .string()
      .min(1)
      .max(100)
      .regex(
        /^[a-zA-Z0-9_-]+$/,
        "Group name can only contain letters, numbers, hyphens, and underscores (no spaces)"
      )
  ),
  description: zfd.text(z.string().optional()),
  isPublic: zfd.checkbox(),
  institutionId: zfd.text(z.string()),
});

export const createGroupAction = actionClient
  .inputSchema(inputSchema)
  .action(async ({ parsedInput: input }) => {
    const session = await getSessionOrRedirect();

    // Check if user is a member of the institution
    const userInstitution = await prisma.userInstitution.findUnique({
      where: {
        userId_institutionId: {
          userId: session.user.id,
          institutionId: input.institutionId,
        },
      },
    });

    if (!userInstitution || !userInstitution.isActive) {
      throw new Error("You must be a member of the institution to create groups");
    }

    // Check if group with this name already exists in this institution
    const existingGroup = await prisma.group.findUnique({
      where: { 
        name_institutionId: {
          name: input.name,
          institutionId: input.institutionId,
        }
      },
    });

    if (existingGroup) {
      throw new Error("A group with this name already exists in this institution");
    }

    // Create the group
    const group = await prisma.group.create({
      data: {
        name: input.name,
        description: input.description || null,
        isPublic: input.isPublic ?? true,
        createdByUserId: session.user.id,
        institutionId: input.institutionId,
      },
      select: {
        id: true,
        name: true,
      },
    });

    // Automatically add the creator as an admin of the group
    await prisma.userGroup.create({
      data: {
        userId: session.user.id,
        groupId: group.id,
        role: "admin",
        isActive: true,
      },
    });

    revalidatePath("/groups");

    return {
      id: group.id,
      name: group.name,
      message: "Group created successfully",
    };
  });
