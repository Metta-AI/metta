"use server";

import { revalidatePath } from "next/cache";
import { zfd } from "zod-form-data";
import { z } from "zod/v4";

import { actionClient } from "@/lib/actionClient";
import { getSessionOrRedirect } from "@/lib/auth";
import { db } from "@/lib/db";
import { postsTable } from "@/lib/db/schema/post";

const inputSchema = zfd.formData({
  title: zfd.text(z.string().min(1).max(255)),
});

export const createPostAction = actionClient
  .inputSchema(inputSchema)
  .action(async ({ parsedInput: input }) => {
    const session = await getSessionOrRedirect();

    const post = await db.insert(postsTable).values({
      title: input.title,
      authorId: session.user.id,
    });

    revalidatePath("/");

    return { id: post.id };
  });
