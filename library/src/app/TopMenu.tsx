import { redirect } from "next/navigation";
import { FC } from "react";
import Link from "next/link";

import { auth } from "@/lib/auth";
import { isGlobalAdmin } from "@/lib/adminAuth";
import { ClientNotificationBell } from "@/components/ClientNotificationBell";
import { MobileMenuButton } from "@/components/MobileMenuButton";

const UserInfo: FC = async () => {
  const session = await auth();

  if (!session?.user?.email) {
    // usually shouldn't happen, we check auth in top-level layout
    redirect("/api/auth/signin");
  }

  return (
    <div className="flex items-center gap-2">
      <ClientNotificationBell />
      <span>{session.user.email}</span>
    </div>
  );
};

export const TopMenu: FC = async () => {
  const session = await auth();
  const isAdmin = await isGlobalAdmin();

  return (
    <div className="flex items-center justify-between border-b border-gray-200 bg-gray-50 px-4 py-2 md:pr-8 md:pl-4">
      {/* Mobile menu button, logo, and admin link */}
      <div className="flex items-center gap-4">
        <MobileMenuButton />

        {/* Logo - visible on desktop */}
        <Link href="/" className="hidden items-center gap-2 md:flex">
          <svg
            className="h-5 w-5 text-gray-500"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            suppressHydrationWarning
          >
            <path d="M3 18h18" />
            <rect x="3" y="8" width="3.6" height="10" />
            <rect x="6.6" y="6" width="3.6" height="12" />
            <rect x="10.2" y="9" width="3.6" height="9" />
            <rect x="13.8" y="7" width="3.6" height="11" />
            <rect x="17.4" y="5" width="3.6" height="13" />
          </svg>
          <span className="text-sm font-semibold text-gray-900">
            Republic of Papers
          </span>
        </Link>

        {isAdmin && (
          <Link
            href="/admin/institutions"
            className="rounded-md bg-red-100 px-2 py-1 text-sm font-medium text-red-700 hover:bg-red-200"
            title="Admin Panel"
          >
            🛠️ Admin
          </Link>
        )}
      </div>
      <UserInfo />
    </div>
  );
};
