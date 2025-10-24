"use server";

import { revalidatePath } from "next/cache";
import { zfd } from "zod-form-data";
import { z } from "zod/v4";

import { actionClient } from "@/lib/actionClient";
import { getSessionOrRedirect } from "@/lib/auth";
import { prisma } from "@/lib/db/prisma";

const inputSchema = zfd.formData({
  paperId: zfd.text(z.string().min(1)),
});

export const toggleStarAction = actionClient
  .inputSchema(inputSchema)
  .action(async ({ parsedInput: input }) => {
    const session = await getSessionOrRedirect();

    // Check if user already has an interaction record for this paper
    const existingInteraction = await prisma.userPaperInteraction.findUnique({
      where: {
        userId_paperId: {
          userId: session.user.id,
          paperId: input.paperId,
        },
      },
    });

    if (existingInteraction) {
      // Update existing interaction
      const currentStarred = existingInteraction.starred;
      const newStarred = !currentStarred;

      await prisma.$transaction([
        prisma.userPaperInteraction.update({
          where: {
            userId_paperId: {
              userId: session.user.id,
              paperId: input.paperId,
            },
          },
          data: {
            starred: newStarred,
          },
        }),
        // Update the paper's star count
        prisma.paper.update({
          where: { id: input.paperId },
          data: {
            stars: {
              increment: newStarred ? 1 : -1,
            },
          },
        }),
      ]);
    } else {
      // Create new interaction record with starred = true
      await prisma.$transaction([
        prisma.userPaperInteraction.create({
          data: {
            userId: session.user.id,
            paperId: input.paperId,
            starred: true,
          },
        }),
        // Increment the paper's star count
        prisma.paper.update({
          where: { id: input.paperId },
          data: {
            stars: {
              increment: 1,
            },
          },
        }),
      ]);
    }

    // Revalidate multiple pages to show updated state
    revalidatePath("/papers");
    revalidatePath("/"); // Front page
    revalidatePath("/posts", "layout"); // All post pages
    revalidatePath("/", "layout"); // Revalidate entire app to catch all pages

    return { success: true };
  });
