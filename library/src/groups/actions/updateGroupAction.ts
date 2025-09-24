"use server";

import { revalidatePath } from "next/cache";
import { zfd } from "zod-form-data";
import { z } from "zod/v4";

import { actionClient } from "@/lib/actionClient";
import { getSessionOrRedirect } from "@/lib/auth";
import { prisma } from "@/lib/db/prisma";

const inputSchema = zfd.formData({
  groupId: zfd.text(z.string()),
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
});

export const updateGroupAction = actionClient
  .inputSchema(inputSchema)
  .action(async ({ parsedInput: input }) => {
    const session = await getSessionOrRedirect();

    // Check if current user has admin rights for this group
    const userGroup = await prisma.userGroup.findUnique({
      where: {
        userId_groupId: {
          userId: session.user.id,
          groupId: input.groupId,
        },
      },
    });

    if (!userGroup || userGroup.role !== "admin") {
      throw new Error("You don't have permission to update this group");
    }

    // Check if another group with this name already exists in the same institution
    if (input.name) {
      const currentGroup = await prisma.group.findUnique({
        where: { id: input.groupId },
        select: { institutionId: true },
      });

      if (!currentGroup) {
        throw new Error("Group not found");
      }

      const existingGroup = await prisma.group.findFirst({
        where: {
          name: input.name,
          institutionId: currentGroup.institutionId,
          id: { not: input.groupId },
        },
      });

      if (existingGroup) {
        throw new Error(
          "A group with this name already exists in this institution"
        );
      }
    }

    // Update the group
    const group = await prisma.group.update({
      where: { id: input.groupId },
      data: {
        name: input.name,
        description: input.description || null,
        isPublic: input.isPublic ?? true,
      },
      select: {
        id: true,
        name: true,
      },
    });

    revalidatePath("/groups");

    return {
      id: group.id,
      name: group.name,
      message: "Group updated successfully",
    };
  });
