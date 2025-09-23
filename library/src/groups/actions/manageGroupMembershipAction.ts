"use server";

import { revalidatePath } from "next/cache";
import { zfd } from "zod-form-data";
import { z } from "zod/v4";

import { actionClient } from "@/lib/actionClient";
import { getSessionOrRedirect } from "@/lib/auth";
import { prisma } from "@/lib/db/prisma";

const inputSchema = zfd.formData({
  groupId: zfd.text(z.string()),
  userEmail: zfd.text(z.string().email()),
  role: zfd.text(z.string().optional()),
  action: zfd.text(z.enum(["add", "update", "remove"])),
});

export const manageGroupMembershipAction = actionClient
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
      throw new Error(
        "You don't have permission to manage users for this group"
      );
    }

    // Find the target user by email
    const targetUser = await prisma.user.findUnique({
      where: { email: input.userEmail },
    });

    if (!targetUser) {
      throw new Error("User not found");
    }

    let result: any = {};

    switch (input.action) {
      case "add":
        // Check if user is already a member
        const existingMembership = await prisma.userGroup.findUnique({
          where: {
            userId_groupId: {
              userId: targetUser.id,
              groupId: input.groupId,
            },
          },
        });

        if (existingMembership) {
          throw new Error("User is already a member of this group");
        }

        result = await prisma.userGroup.create({
          data: {
            userId: targetUser.id,
            groupId: input.groupId,
            role: input.role || "member",
            isActive: true,
          },
          include: {
            user: {
              select: { name: true, email: true },
            },
          },
        });
        break;

      case "update":
        result = await prisma.userGroup.update({
          where: {
            userId_groupId: {
              userId: targetUser.id,
              groupId: input.groupId,
            },
          },
          data: {
            role: input.role,
          },
          include: {
            user: {
              select: { name: true, email: true },
            },
          },
        });
        break;

      case "remove":
        // Prevent removing the last admin
        const adminCount = await prisma.userGroup.count({
          where: {
            groupId: input.groupId,
            role: "admin",
            isActive: true,
          },
        });

        const targetMembership = await prisma.userGroup.findUnique({
          where: {
            userId_groupId: {
              userId: targetUser.id,
              groupId: input.groupId,
            },
          },
        });

        if (targetMembership?.role === "admin" && adminCount <= 1) {
          throw new Error("Cannot remove the last admin of the group");
        }

        await prisma.userGroup.delete({
          where: {
            userId_groupId: {
              userId: targetUser.id,
              groupId: input.groupId,
            },
          },
        });

        result = { message: "User removed from group" };
        break;
    }

    revalidatePath("/groups");

    return {
      ...result,
      message: `User ${input.action === "add" ? "added to" : input.action === "update" ? "updated in" : "removed from"} group successfully`,
    };
  });
