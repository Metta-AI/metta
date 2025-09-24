"use server";

import { revalidatePath } from "next/cache";
import { zfd } from "zod-form-data";
import { z } from "zod/v4";

import { actionClient } from "@/lib/actionClient";
import { getSessionOrRedirect } from "@/lib/auth";
import { prisma } from "@/lib/db/prisma";

const inputSchema = zfd.formData({
  paperId: zfd.text(z.string().min(1)),
  postId: zfd.text(z.string().min(1)),
});

export const toggleQueueAction = actionClient
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

    let isNowQueued = false;

    if (existingInteraction) {
      // Update existing interaction
      const currentQueued = existingInteraction.queued;
      isNowQueued = !currentQueued;
      await prisma.userPaperInteraction.update({
        where: {
          userId_paperId: {
            userId: session.user.id,
            paperId: input.paperId,
          },
        },
        data: {
          queued: isNowQueued,
        },
      });
    } else {
      // Create new interaction record with queued = true
      isNowQueued = true;
      await prisma.userPaperInteraction.create({
        data: {
          userId: session.user.id,
          paperId: input.paperId,
          queued: true,
        },
      });
    }

    // Update the post's queue count
    await prisma.post.update({
      where: {
        id: input.postId,
      },
      data: {
        queues: {
          increment: isNowQueued ? 1 : -1,
        },
      },
    });

    // Revalidate pages to show updated state
    revalidatePath("/papers");
    revalidatePath("/"); // Revalidate main feed page

    return { success: true };
  });
