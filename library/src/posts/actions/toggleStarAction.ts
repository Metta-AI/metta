"use server";

import { revalidatePath } from "next/cache";
import { zfd } from "zod-form-data";
import { z } from "zod/v4";

import { actionClient } from "@/lib/actionClient";
import { getSessionOrRedirect } from "@/lib/auth";
import { db } from "@/lib/db";
import { userPaperInteractionsTable } from "@/lib/db/schema/paper";
import { eq, and } from "drizzle-orm";

const inputSchema = zfd.formData({
  paperId: zfd.text(z.string().min(1)),
});

export const toggleStarAction = actionClient
  .inputSchema(inputSchema)
  .action(async ({ parsedInput: input }) => {
    const session = await getSessionOrRedirect();

    // Check if user already has an interaction record for this paper
    const existingInteraction = await db
      .select()
      .from(userPaperInteractionsTable)
      .where(
        and(
          eq(userPaperInteractionsTable.userId, session.user.id),
          eq(userPaperInteractionsTable.paperId, input.paperId)
        )
      )
      .limit(1);

    if (existingInteraction.length > 0) {
      // Update existing interaction
      const currentStarred = existingInteraction[0].starred;
      await db
        .update(userPaperInteractionsTable)
        .set({ starred: !currentStarred })
        .where(
          and(
            eq(userPaperInteractionsTable.userId, session.user.id),
            eq(userPaperInteractionsTable.paperId, input.paperId)
          )
        );
    } else {
      // Create new interaction record with starred = true
      await db.insert(userPaperInteractionsTable).values({
        userId: session.user.id,
        paperId: input.paperId,
        starred: true,
      });
    }

    // Revalidate the papers page to show updated state
    revalidatePath("/papers");

    return { success: true };
  }); 