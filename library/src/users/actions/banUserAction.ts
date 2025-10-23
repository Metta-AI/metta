"use server";

import { revalidatePath } from "next/cache";
import { zfd } from "zod-form-data";
import { z } from "zod/v4";

import { actionClient, ActionError } from "@/lib/actionClient";
import { getAdminSessionOrRedirect } from "@/lib/adminAuth";
import { prisma } from "@/lib/db/prisma";

const inputSchema = zfd.formData({
  userEmail: zfd.text(z.string().email()),
  reason: zfd.text(z.string().min(1).max(500)),
});

export const banUserAction = actionClient
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
      },
    });

    if (!targetUser) {
      throw new ActionError("User not found");
    }

    if (targetUser.isBanned) {
      throw new ActionError("User is already banned");
    }

    // Prevent admins from banning themselves
    if (targetUser.id === session.user.id) {
      throw new ActionError("You cannot ban yourself");
    }

    // Ban the user
    await prisma.user.update({
      where: { id: targetUser.id },
      data: {
        isBanned: true,
        bannedAt: new Date(),
        banReason: input.reason,
        bannedByUserId: session.user.id,
      },
    });

    // Revalidate admin pages
    revalidatePath("/admin");
    revalidatePath("/admin/users");

    return {
      success: true,
      message: `User ${targetUser.name || targetUser.email} has been banned.`,
      userId: targetUser.id,
      userEmail: targetUser.email,
    };
  });
