"use server";

import { revalidatePath } from "next/cache";
import { zfd } from "zod-form-data";
import { z } from "zod/v4";

import { actionClient } from "@/lib/actionClient";
import { getSessionOrRedirect } from "@/lib/auth";
import { prisma } from "@/lib/db/prisma";

const inputSchema = zfd.formData({
  postId: zfd.text(z.string().min(1)),
  parentId: zfd.text(z.string().optional()),
  content: zfd.text(
    z
      .string()
      .min(1, "Comment cannot be empty")
      .max(2000, "Comment is too long")
  ),
});

export const createCommentAction = actionClient
  .inputSchema(inputSchema)
  .action(async ({ parsedInput: input }) => {
    const session = await getSessionOrRedirect();

    if (!session?.user?.id) {
      throw new Error("You must be signed in to comment");
    }

    // Validate parent comment exists if parentId is provided
    if (input.parentId) {
      const parentComment = await prisma.comment.findUnique({
        where: { id: input.parentId },
        select: { id: true, postId: true },
      });

      if (!parentComment) {
        throw new Error("Parent comment not found");
      }

      if (parentComment.postId !== input.postId) {
        throw new Error("Parent comment must belong to the same post");
      }
    }

    // Create the comment
    const comment = await prisma.comment.create({
      data: {
        content: input.content.trim(),
        postId: input.postId,
        parentId: input.parentId || null,
        authorId: session.user.id,
      },
      include: {
        author: true,
      },
    });

    // Update the post's reply count
    await prisma.post.update({
      where: { id: input.postId },
      data: {
        replies: {
          increment: 1,
        },
      },
    });

    // Revalidate the current page to show updated data
    revalidatePath("/");

    return {
      success: true,
      comment: {
        id: comment.id,
        content: comment.content,
        postId: comment.postId,
        parentId: comment.parentId,
        author: {
          id: comment.author.id,
          name: comment.author.name,
          email: comment.author.email,
          image: comment.author.image,
        },
        createdAt: comment.createdAt,
        updatedAt: comment.updatedAt,
      },
    };
  });
