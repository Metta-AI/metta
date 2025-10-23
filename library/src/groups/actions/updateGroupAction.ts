"use server";

import { revalidatePath } from "next/cache";
import { zfd } from "zod-form-data";
import { z } from "zod/v4";

import { actionClient } from "@/lib/actionClient";
import { getSessionOrRedirect } from "@/lib/auth";
import { prisma } from "@/lib/db/prisma";
import { validateGroupName } from "@/lib/name-validation";
import { GroupRepository } from "../data/group-repository";
import { AuthorizationError, ConflictError, NotFoundError } from "@/lib/errors";

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
    const isAdmin = await GroupRepository.isAdmin(
      session.user.id,
      input.groupId
    );

    if (!isAdmin) {
      throw new AuthorizationError(
        "You don't have permission to update this group"
      );
    }

    // Check if another group with this name already exists in the same institution
    if (input.name) {
      const currentGroup = await GroupRepository.findById(input.groupId);

      if (!currentGroup) {
        throw new NotFoundError("Group", input.groupId);
      }

      const existingGroup = await prisma.group.findFirst({
        where: {
          name: input.name,
          institutionId: currentGroup.institutionId,
          id: { not: input.groupId },
        },
      });

      if (existingGroup) {
        throw new ConflictError(
          "A group with this name already exists in this institution"
        );
      }

      // Validate name uniqueness across all entity types (excluding current group)
      await validateGroupName(input.name, input.groupId);
    }

    // Update the group
    const group = await GroupRepository.update(input.groupId, {
      name: input.name,
      description: input.description || null,
      isPublic: input.isPublic ?? true,
    });

    revalidatePath("/groups");

    return {
      id: group.id,
      name: group.name,
      message: "Group updated successfully",
    };
  });
