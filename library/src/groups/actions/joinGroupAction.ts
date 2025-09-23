"use server";

import { revalidatePath } from "next/cache";
import { zfd } from "zod-form-data";
import { z } from "zod/v4";

import { actionClient } from "@/lib/actionClient";
import { getSessionOrRedirect } from "@/lib/auth";
import { prisma } from "@/lib/db/prisma";

const inputSchema = zfd.formData({
  groupId: zfd.text(z.string()),
});

export const joinGroupAction = actionClient
  .inputSchema(inputSchema)
  .action(async ({ parsedInput: input }) => {
    const session = await getSessionOrRedirect();

    // Get the group to check institution and public status
    const group = await prisma.group.findUnique({
      where: { id: input.groupId },
      select: {
        id: true,
        name: true,
        isPublic: true,
        institutionId: true,
      },
    });

    if (!group) {
      throw new Error("Group not found");
    }

    // Check if group is public
    if (!group.isPublic) {
      throw new Error("This group is private and requires an invitation");
    }

    // Check if user is a member of the same institution
    const userInstitution = await prisma.userInstitution.findUnique({
      where: {
        userId_institutionId: {
          userId: session.user.id,
          institutionId: group.institutionId,
        },
      },
    });

    if (!userInstitution || !userInstitution.isActive) {
      throw new Error(
        "You must be a member of the same institution to join this group"
      );
    }

    // Check if user is already a member
    const existingMembership = await prisma.userGroup.findUnique({
      where: {
        userId_groupId: {
          userId: session.user.id,
          groupId: input.groupId,
        },
      },
    });

    if (existingMembership) {
      throw new Error("You are already a member of this group");
    }

    // Add user to the group as a member
    await prisma.userGroup.create({
      data: {
        userId: session.user.id,
        groupId: input.groupId,
        role: "member",
        isActive: true,
      },
    });

    revalidatePath("/groups");

    return {
      groupId: group.id,
      groupName: group.name,
      message: `Successfully joined ${group.name}`,
    };
  });
