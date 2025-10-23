"use server";

import { revalidatePath } from "next/cache";
import { zfd } from "zod-form-data";
import { z } from "zod/v4";

import { actionClient } from "@/lib/actionClient";
import { getSessionOrRedirect } from "@/lib/auth";
import { GroupMembershipService } from "../services/membership-service";
import { GroupRepository } from "../data/group-repository";

const inputSchema = zfd.formData({
  groupId: zfd.text(z.string()),
});

export const joinGroupAction = actionClient
  .inputSchema(inputSchema)
  .action(async ({ parsedInput: input }) => {
    const session = await getSessionOrRedirect();

    // Use service to handle join logic
    const result = await GroupMembershipService.joinGroup(
      session.user.id,
      input.groupId
    );

    // Get group name for response message
    const group = await GroupRepository.findByIdWithBasicInfo(input.groupId);

    revalidatePath("/groups");

    return {
      groupId: group?.id,
      groupName: group?.name,
      message: `Successfully joined ${group?.name}`,
    };
  });
