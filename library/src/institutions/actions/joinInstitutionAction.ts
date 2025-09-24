"use server";

import { revalidatePath } from "next/cache";
import { zfd } from "zod-form-data";
import { z } from "zod/v4";

import { actionClient } from "@/lib/actionClient";
import { getSessionOrRedirect } from "@/lib/auth";
import { prisma } from "@/lib/db/prisma";

const inputSchema = zfd.formData({
  institutionId: zfd.text(z.string()),
  role: zfd.text(z.string().optional()),
  department: zfd.text(z.string().optional()),
  title: zfd.text(z.string().optional()),
});

export const joinInstitutionAction = actionClient
  .inputSchema(inputSchema)
  .action(async ({ parsedInput: input }) => {
    const session = await getSessionOrRedirect();

    // Get the institution details
    const institution = await prisma.institution.findUnique({
      where: { id: input.institutionId },
      select: {
        id: true,
        name: true,
        domain: true,
        requiresApproval: true,
      },
    });

    if (!institution) {
      throw new Error("Institution not found");
    }

    // Check if user is already a member
    const existingMembership = await prisma.userInstitution.findUnique({
      where: {
        userId_institutionId: {
          userId: session.user.id,
          institutionId: input.institutionId,
        },
      },
    });

    if (existingMembership) {
      if (existingMembership.status === "PENDING") {
        throw new Error(
          "Your request to join this institution is already pending approval"
        );
      } else if (existingMembership.status === "REJECTED") {
        throw new Error(
          "Your request to join this institution was previously rejected"
        );
      } else {
        throw new Error("You are already a member of this institution");
      }
    }

    // Check if user's email domain matches institution domain (auto-approve if match)
    const userEmail = session.user.email;
    let domainAutoApproved = false;

    if (userEmail && institution.domain) {
      const userDomain = userEmail.split("@")[1];
      if (
        userDomain &&
        userDomain.toLowerCase() === institution.domain.toLowerCase()
      ) {
        domainAutoApproved = true;
      }
    }

    // Determine membership status based on approval requirements
    const requiresApproval =
      institution.requiresApproval && !domainAutoApproved;
    const status = requiresApproval ? "PENDING" : "APPROVED";
    const isActive = !requiresApproval; // Only active if immediately approved

    // Create the membership
    await prisma.userInstitution.create({
      data: {
        userId: session.user.id,
        institutionId: input.institutionId,
        role: input.role || "member",
        department: input.department || null,
        title: input.title || null,
        status: status,
        isActive: isActive,
      },
    });

    revalidatePath("/institutions");

    // Return appropriate message based on status
    if (requiresApproval) {
      return {
        success: true,
        status: "pending",
        message: `Your request to join ${institution.name} has been submitted and is pending approval from an administrator`,
        institutionName: institution.name,
      };
    } else {
      return {
        success: true,
        status: "approved",
        message: domainAutoApproved
          ? `Successfully joined ${institution.name} (auto-approved via email domain)`
          : `Successfully joined ${institution.name}`,
        institutionName: institution.name,
      };
    }
  });
