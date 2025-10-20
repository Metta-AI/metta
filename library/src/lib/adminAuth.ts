import "server-only";

import { auth } from "@/lib/auth";
import { prisma } from "@/lib/db/prisma";
import { redirect } from "next/navigation";

/**
 * Bootstrap admin emails from environment variables
 * Format: ADMIN_EMAILS=email1@example.com,email2@example.com
 */
const BOOTSTRAP_ADMIN_EMAILS = process.env.ADMIN_EMAILS
  ? process.env.ADMIN_EMAILS.split(",").map((email) => email.trim())
  : [];

/**
 * Check if the current user is a global admin
 *
 * Checks both:
 * 1. Database isAdmin flag (preferred, can be managed via UI)
 * 2. Bootstrap admin emails from env vars (for initial setup)
 */
export async function isGlobalAdmin(): Promise<boolean> {
  const session = await auth();
  if (!session?.user?.email) return false;

  // Check database flag first
  const user = await prisma.user.findUnique({
    where: { email: session.user.email },
    select: { isAdmin: true },
  });

  if (user?.isAdmin) return true;

  // Fallback to bootstrap admin emails
  return BOOTSTRAP_ADMIN_EMAILS.includes(session.user.email);
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
