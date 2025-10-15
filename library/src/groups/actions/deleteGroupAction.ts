"use server";

import { revalidatePath } from "next/cache";
import { zfd } from "zod-form-data";
import { z } from "zod/v4";

import { actionClient } from "@/lib/actionClient";
import { getSessionOrRedirect } from "@/lib/auth";
import { AuthorizationError, NotFoundError } from "@/lib/errors";
import { GroupRepository } from "@/groups/data/group-repository";

const inputSchema = zfd.formData({
  groupId: zfd.text(z.string().min(1)),
});

export const deleteGroupAction = actionClient
  .inputSchema(inputSchema)
  .action(async ({ parsedInput: input }) => {
    const session = await getSessionOrRedirect();

    // Check if current user has admin rights for this group
    const isAdmin = await GroupRepository.isAdmin(
      session.user.id,
      input.groupId
    );

    if (!isAdmin) {
      throw new AuthorizationError(
        "You don't have permission to delete this group"
      );
    }

    // Verify group exists
    const group = await GroupRepository.findById(input.groupId);
    if (!group) {
      throw new NotFoundError("Group", input.groupId);
    }

    // Delete the group (memberships will be cascade deleted)
    await GroupRepository.delete(input.groupId);

    revalidatePath("/groups");

    return {
      success: true,
      message: "Group deleted successfully",
    };
  });
