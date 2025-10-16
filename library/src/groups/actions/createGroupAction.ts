"use server";

import { revalidatePath } from "next/cache";
import { zfd } from "zod-form-data";
import { z } from "zod/v4";

import { actionClient } from "@/lib/actionClient";
import { getSessionOrRedirect } from "@/lib/auth";
import { prisma } from "@/lib/db/prisma";
import { validateGroupName } from "@/lib/name-validation";
import { GroupRepository } from "../data/group-repository";
import { AuthorizationError, ConflictError } from "@/lib/errors";

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
    const isInInstitution = await GroupRepository.isUserInInstitution(
      session.user.id,
      input.institutionId
    );

    if (!isInInstitution) {
      throw new AuthorizationError(
        "You must be a member of the institution to create groups"
      );
    }

    // Check if group with this name already exists in this institution
    const existingGroup = await prisma.group.findUnique({
      where: {
        name_institutionId: {
          name: input.name,
          institutionId: input.institutionId,
        },
      },
    });

    if (existingGroup) {
      throw new ConflictError(
        "A group with this name already exists in this institution"
      );
    }

    // Validate name uniqueness across all entity types (for mentions)
    await validateGroupName(input.name);

    // Create the group
    const group = await GroupRepository.create({
      name: input.name,
      description: input.description || null,
      isPublic: input.isPublic ?? true,
      createdByUserId: session.user.id,
      institutionId: input.institutionId,
    });

    // Automatically add the creator as an admin of the group
    await GroupRepository.createMembership({
      userId: session.user.id,
      groupId: group.id,
      role: "admin",
      isActive: true,
    });

    revalidatePath("/groups");

    return {
      id: group.id,
      name: group.name,
      message: "Group created successfully",
    };
  });
