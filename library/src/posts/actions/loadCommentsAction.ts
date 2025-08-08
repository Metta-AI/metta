"use server";

import { revalidatePath } from "next/cache";
import { zfd } from "zod-form-data";
import { z } from "zod/v4";

import { actionClient } from "@/lib/actionClient";
import { prisma } from "@/lib/db/prisma";

const inputSchema = zfd.formData({
  postId: zfd.text(z.string().min(1)),
});

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

    // Transform to DTO format
    return comments.map((comment) => ({
      id: comment.id,
      content: comment.content,
      postId: comment.postId,
      author: {
        id: comment.author.id,
        name: comment.author.name,
        email: comment.author.email,
        image: comment.author.image,
      },
      createdAt: comment.createdAt,
      updatedAt: comment.updatedAt,
    }));
  });
