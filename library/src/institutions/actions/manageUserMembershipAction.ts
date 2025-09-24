"use server";

import { revalidatePath } from "next/cache";
import { zfd } from "zod-form-data";
import { z } from "zod/v4";

import { actionClient } from "@/lib/actionClient";
import { getSessionOrRedirect } from "@/lib/auth";
import { prisma } from "@/lib/db/prisma";

const inputSchema = zfd.formData({
  institutionId: zfd.text(z.string()),
  userEmail: zfd.text(z.string().email()),
  role: zfd.text(z.string().optional()),
  department: zfd.text(z.string().optional()),
  title: zfd.text(z.string().optional()),
  action: zfd.text(z.enum(["add", "update", "remove"])),
});

export const manageUserMembershipAction = actionClient
  .inputSchema(inputSchema)
  .action(async ({ parsedInput: input }) => {
    const session = await getSessionOrRedirect();

    // Check if current user has admin or owner rights for this institution
    const userInstitution = await prisma.userInstitution.findUnique({
      where: {
        userId_institutionId: {
          userId: session.user.id,
          institutionId: input.institutionId,
        },
      },
    });

    if (!userInstitution || userInstitution.role !== "admin") {
      throw new Error(
        "You don't have permission to manage users for this institution"
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
        const existingMembership = await prisma.userInstitution.findUnique({
          where: {
            userId_institutionId: {
              userId: targetUser.id,
              institutionId: input.institutionId,
            },
          },
        });

        if (existingMembership) {
          throw new Error("User is already a member of this institution");
        }

        result = await prisma.userInstitution.create({
          data: {
            userId: targetUser.id,
            institutionId: input.institutionId,
            role: input.role || "member",
            department: input.department || null,
            title: input.title || null,
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
        result = await prisma.userInstitution.update({
          where: {
            userId_institutionId: {
              userId: targetUser.id,
              institutionId: input.institutionId,
            },
          },
          data: {
            role: input.role,
            department: input.department || null,
            title: input.title || null,
          },
          include: {
            user: {
              select: { name: true, email: true },
            },
          },
        });
        break;

      case "remove":
        const targetMembership = await prisma.userInstitution.findUnique({
          where: {
            userId_institutionId: {
              userId: targetUser.id,
              institutionId: input.institutionId,
            },
          },
        });

        // Prevent removing the last admin
        if (targetMembership?.role === "admin") {
          const adminCount = await prisma.userInstitution.count({
            where: {
              institutionId: input.institutionId,
              role: "admin",
              isActive: true,
            },
          });

          if (adminCount <= 1) {
            throw new Error("Cannot remove the last admin of the institution");
          }
        }

        await prisma.userInstitution.delete({
          where: {
            userId_institutionId: {
              userId: targetUser.id,
              institutionId: input.institutionId,
            },
          },
        });

        result = { message: "User removed from institution" };
        break;
    }

    revalidatePath("/institutions");

    return {
      ...result,
      message: `User ${input.action === "add" ? "added to" : input.action === "update" ? "updated in" : "removed from"} institution successfully`,
    };
  });
