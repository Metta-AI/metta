import "server-only";

import { auth } from "@/lib/auth";
import { redirect } from "next/navigation";

/**
 * List of email addresses that have admin access
 * In production, this could be moved to environment variables or database
 */
const ADMIN_EMAILS = [
  // Add admin email addresses here
  "admin@softmax.com",
  "admin@stem.ai",
];

/**
 * Check if the current user is a global admin
 */
export async function isGlobalAdmin(): Promise<boolean> {
  const session = await auth();
  if (!session?.user?.email) return false;

  return ADMIN_EMAILS.includes(session.user.email);
}

/**
 * Get current session and verify admin access, redirect if not authorized
 */
export async function getAdminSessionOrRedirect() {
  const session = await auth();

  if (!session?.user) {
    redirect("/api/auth/signin");
  }

  if (!(await isGlobalAdmin())) {
    redirect("/");
  }

  return session;
}

/**
 * Check if user has admin access to a specific institution
 * This includes both global admins and institution-specific admins
 */
export async function hasInstitutionAdminAccess(
  institutionId: string
): Promise<boolean> {
  const session = await auth();
  if (!session?.user) return false;

  // Global admins have access to everything
  if (await isGlobalAdmin()) return true;

  // Check if user is admin of the institution
  const { prisma } = await import("@/lib/db/prisma");
  const userInstitution = await prisma.userInstitution.findUnique({
    where: {
      userId_institutionId: {
        userId: session.user.id!,
        institutionId,
      },
    },
  });

  return userInstitution?.role === "admin";
}
