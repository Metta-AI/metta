"use server";

import { z } from "zod/v4";

import { actionClient } from "@/lib/actionClient";
import { db } from "@/lib/db";
import { postsTable } from "@/lib/db/schema/post";
import { getSessionOrRedirect } from "@/lib/auth";

const schema = z.object({
  title: z.string(),
});

export const createPostAction = actionClient
  .schema(schema)
  .action(async ({ parsedInput: input }) => {
    const session = await getSessionOrRedirect();

    const post = await db.insert(postsTable).values({
      title: input.title,
      authorId: session.user.id,
    });

    return { id: post.id };
  });
