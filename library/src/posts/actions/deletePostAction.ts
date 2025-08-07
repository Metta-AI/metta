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

export const deletePostAction = actionClient
  .inputSchema(inputSchema)
  .action(async ({ parsedInput: input }) => {
    const session = await getSessionOrRedirect();

    // Get the post and verify ownership
    const post = await prisma.post.findUnique({
      where: { id: input.postId },
      select: { id: true, authorId: true },
    });

    if (!post) {
      throw new Error("Post not found");
    }

    if (post.authorId !== session.user.id) {
      throw new Error("You can only delete your own posts");
    }

    // Delete the post
    await prisma.post.delete({
      where: { id: input.postId },
    });

    revalidatePath("/");

    return { success: true };
  });
