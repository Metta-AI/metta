"use server";

import { revalidatePath } from "next/cache";
import { zfd } from "zod-form-data";
import { z } from "zod/v4";

import { actionClient } from "@/lib/actionClient";
import { getSessionOrRedirect } from "@/lib/auth";
import { prisma } from "@/lib/db/prisma";

const inputSchema = zfd.formData({
  title: zfd.text(z.string().min(1).max(255)),
  content: zfd.text(z.string().optional()),
  postType: zfd.text(z.enum(['user-post', 'paper-post', 'pure-paper']).optional()),
  paperId: zfd.text(z.string().optional()), // Added support for paperId
});

export const createPostAction = actionClient
  .inputSchema(inputSchema)
  .action(async ({ parsedInput: input }) => {
    const session = await getSessionOrRedirect();

    const post = await prisma.post.create({
      data: {
        title: input.title,
        content: input.content || null,
        postType: input.postType || 'user-post',
        paperId: input.paperId || null, // Added paperId support
        authorId: session.user.id,
      },
      select: {
        id: true,
      },
    });

    revalidatePath("/");

    return { id: post.id };
  }); 