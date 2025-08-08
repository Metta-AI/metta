"use server";

import { revalidatePath } from "next/cache";
import { zfd } from "zod-form-data";
import { z } from "zod/v4";

import { actionClient } from "@/lib/actionClient";
import { getSessionOrRedirect } from "@/lib/auth";
import { prisma } from "@/lib/db/prisma";

const inputSchema = zfd.formData({
  postId: zfd.text(z.string().min(1)),
});

export const toggleLikeAction = actionClient
  .inputSchema(inputSchema)
  .action(async ({ parsedInput: input }) => {
    const session = await getSessionOrRedirect();

    // Check if user already has a like record for this post
    const existingLike = await prisma.userPostLike.findUnique({
      where: {
        userId_postId: {
          userId: session.user.id,
          postId: input.postId,
        },
      },
    });

    if (existingLike) {
      // Remove the like
      await prisma.userPostLike.delete({
        where: {
          userId_postId: {
            userId: session.user.id,
            postId: input.postId,
          },
        },
      });

      // Decrement the likes count on the post
      await prisma.post.update({
        where: { id: input.postId },
        data: {
          likes: {
            decrement: 1,
          },
        },
      });
    } else {
      // Create new like record
      await prisma.userPostLike.create({
        data: {
          userId: session.user.id,
          postId: input.postId,
        },
      });

      // Increment the likes count on the post
      await prisma.post.update({
        where: { id: input.postId },
        data: {
          likes: {
            increment: 1,
          },
        },
      });
    }

    // Revalidate the library page to show updated state
    revalidatePath("/library");
    revalidatePath("/");

    return { success: true };
  });
