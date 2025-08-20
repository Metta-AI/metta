"use server";

import { revalidatePath } from "next/cache";
import { zfd } from "zod-form-data";
import { z } from "zod/v4";

import { actionClient } from "@/lib/actionClient";
import { prisma } from "@/lib/db/prisma";
import { CommentDTO } from "@/posts/data/comments";

const inputSchema = zfd.formData({
  postId: zfd.text(z.string().min(1)),
});

// Helper function to recursively build comment tree
function buildCommentTree(
  comments: any[],
  parentId: string | null = null,
  depth: number = 0
): CommentDTO[] {
  const children = comments.filter((comment) => comment.parentId === parentId);

  return children.map((comment) => ({
    id: comment.id,
    content: comment.content,
    postId: comment.postId,
    parentId: comment.parentId,
    isBot: comment.isBot,
    botType: comment.botType,
    author: {
      id: comment.author.id,
      name: comment.author.name,
      email: comment.author.email,
      image: comment.author.image,
    },
    createdAt: comment.createdAt,
    updatedAt: comment.updatedAt,
    depth,
    replies: buildCommentTree(comments, comment.id, depth + 1),
  }));
}

export const loadCommentsAction = actionClient
  .inputSchema(inputSchema)
  .action(async ({ parsedInput: input }) => {
    const comments = await prisma.comment.findMany({
      where: { postId: input.postId },
      orderBy: { createdAt: "asc" },
      include: {
        author: true,
      },
    });

    // Build hierarchical comment tree
    return buildCommentTree(comments, null);
  });
