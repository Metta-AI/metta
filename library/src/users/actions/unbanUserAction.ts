"use server";

import { revalidatePath } from "next/cache";
import { zfd } from "zod-form-data";
import { z } from "zod/v4";

import { actionClient } from "@/lib/actionClient";
import { getAdminSessionOrRedirect } from "@/lib/adminAuth";
import { prisma } from "@/lib/db/prisma";
import { NotFoundError, ConflictError } from "@/lib/errors";

const inputSchema = zfd.formData({
  userEmail: zfd.text(z.string().email()),
});

export const unbanUserAction = actionClient
  .inputSchema(inputSchema)
  .action(async ({ parsedInput: input }) => {
    // Verify admin access (will throw/redirect if not admin)
    const session = await getAdminSessionOrRedirect();

    // Find the target user by email
    const targetUser = await prisma.user.findUnique({
      where: { email: input.userEmail },
      select: {
        id: true,
        name: true,
        email: true,
        isBanned: true,
        bannedAt: true,
        banReason: true,
      },
    });

    if (!targetUser) {
      throw new NotFoundError("User", input.userEmail);
    }

    if (!targetUser.isBanned) {
      throw new ConflictError("User is not banned");
    }

    // Unban the user
    await prisma.user.update({
      where: { id: targetUser.id },
      data: {
        isBanned: false,
        bannedAt: null,
        banReason: null,
        bannedByUserId: null,
      },
    });

    // Revalidate admin pages
    revalidatePath("/admin");
    revalidatePath("/admin/users");

    return {
      success: true,
      message: `User ${targetUser.name || targetUser.email} has been unbanned.`,
      userId: targetUser.id,
      userEmail: targetUser.email,
    };
  });
