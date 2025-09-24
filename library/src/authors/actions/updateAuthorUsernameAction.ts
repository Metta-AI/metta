"use server";

import { revalidatePath } from "next/cache";
import { zfd } from "zod-form-data";
import { z } from "zod/v4";

import { actionClient } from "@/lib/actionClient";
import { getSessionOrRedirect } from "@/lib/auth";
import { prisma } from "@/lib/db/prisma";
import { validateAuthorUsername } from "@/lib/name-validation";

const inputSchema = zfd.formData({
  authorId: zfd.text(z.string()),
  username: zfd.text(
    z
      .string()
      .min(1)
      .max(50)
      .regex(
        /^[a-zA-Z0-9_-]+$/,
        "Username can only contain letters, numbers, hyphens, and underscores"
      )
  ),
});

export const updateAuthorUsernameAction = actionClient
  .inputSchema(inputSchema)
  .action(async ({ parsedInput: input }) => {
    const session = await getSessionOrRedirect();

    // Find the author
    const author = await prisma.author.findUnique({
      where: { id: input.authorId },
    });

    if (!author) {
      throw new Error("Author not found");
    }

    // TODO: Add permission check - only the author themselves or admins should be able to set usernames
    // This would need to be implemented when author claiming/authentication is added

    // Validate username uniqueness across all entity types
    await validateAuthorUsername(input.username, input.authorId);

    // Update the author's username
    const updatedAuthor = await prisma.author.update({
      where: { id: input.authorId },
      data: {
        username: input.username,
        updatedAt: new Date(),
      },
      select: {
        id: true,
        name: true,
        username: true,
      },
    });

    revalidatePath(`/authors/${input.authorId}`);

    return {
      id: updatedAuthor.id,
      name: updatedAuthor.name,
      username: updatedAuthor.username,
      message: "Author username updated successfully",
    };
  });
