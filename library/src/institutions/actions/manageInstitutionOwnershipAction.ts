"use server";

import { revalidatePath } from "next/cache";
import { zfd } from "zod-form-data";
import { z } from "zod/v4";

import { actionClient } from "@/lib/actionClient";
import { getAdminSessionOrRedirect } from "@/lib/adminAuth";
import { prisma } from "@/lib/db/prisma";

const inputSchema = zfd.formData({
  institutionId: zfd.text(z.string()),
  userEmail: zfd.text(z.string().email()),
  action: zfd.text(z.enum(["assign_admin", "remove_admin"])),
});

export const manageInstitutionOwnershipAction = actionClient
  .inputSchema(inputSchema)
  .action(async ({ parsedInput: input }) => {
    // Verify global admin access
    await getAdminSessionOrRedirect();

    // Find the institution
    const institution = await prisma.institution.findUnique({
      where: { id: input.institutionId },
      select: { id: true, name: true },
    });

    if (!institution) {
      throw new Error("Institution not found");
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
      case "assign_admin":
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
          // Update existing membership to admin
          result = await prisma.userInstitution.update({
            where: {
              userId_institutionId: {
                userId: targetUser.id,
                institutionId: input.institutionId,
              },
            },
            data: {
              role: "admin",
              isActive: true,
            },
            include: {
              user: {
                select: { name: true, email: true },
              },
            },
          });
        } else {
          // Create new admin membership
          result = await prisma.userInstitution.create({
            data: {
              userId: targetUser.id,
              institutionId: input.institutionId,
              role: "admin",
              isActive: true,
            },
            include: {
              user: {
                select: { name: true, email: true },
              },
            },
          });
        }
        break;

      case "remove_admin":
        // Check if this is the last admin
        const adminCount = await prisma.userInstitution.count({
          where: {
            institutionId: input.institutionId,
            role: "admin",
            isActive: true,
          },
        });

        const targetMembership = await prisma.userInstitution.findUnique({
          where: {
            userId_institutionId: {
              userId: targetUser.id,
              institutionId: input.institutionId,
            },
          },
        });

        if (targetMembership?.role === "admin" && adminCount <= 1) {
          throw new Error("Cannot remove the last admin of the institution");
        }

        // Convert to member
        if (targetMembership?.role === "admin") {
          result = await prisma.userInstitution.update({
            where: {
              userId_institutionId: {
                userId: targetUser.id,
                institutionId: input.institutionId,
              },
            },
            data: {
              role: "member",
            },
            include: {
              user: {
                select: { name: true, email: true },
              },
            },
          });
        }
        break;
    }

    revalidatePath("/admin/institutions");

    return {
      ...result,
      message: `Successfully ${input.action.replace("_", " ")} for ${institution.name}`,
    };
  });
