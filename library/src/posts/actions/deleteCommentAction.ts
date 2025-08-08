"use server";

import { revalidatePath } from "next/cache";
import { zfd } from "zod-form-data";
import { z } from "zod/v4";

import { actionClient } from "@/lib/actionClient";
import { getSessionOrRedirect } from "@/lib/auth";
import { prisma } from "@/lib/db/prisma";

const inputSchema = zfd.formData({
  commentId: zfd.text(z.string().min(1)),
});

export const deleteCommentAction = actionClient
  .inputSchema(inputSchema)
  .action(async ({ parsedInput: input }) => {
    const session = await getSessionOrRedirect();

    // Get the comment and verify ownership
    const comment = await prisma.comment.findUnique({
      where: { id: input.commentId },
      include: { post: true },
    });

    if (!comment) {
      throw new Error("Comment not found");
    }

    if (comment.authorId !== session.user.id) {
      throw new Error("You can only delete your own comments");
    }

    // Delete the comment
    await prisma.comment.delete({
      where: { id: input.commentId },
    });

    // Decrement the post's reply count
    await prisma.post.update({
      where: { id: comment.postId },
      data: {
        replies: {
          decrement: 1,
        },
      },
    });

    // Revalidate the current page to show updated data
    revalidatePath("/");

    return { success: true };
  });
