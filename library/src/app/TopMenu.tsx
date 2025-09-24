import { redirect } from "next/navigation";
import { FC } from "react";
import Link from "next/link";

import { auth } from "@/lib/auth";
import { isGlobalAdmin } from "@/lib/adminAuth";
import { ClientNotificationBell } from "@/components/ClientNotificationBell";

import { SignOutButton } from "./SignOutButton";

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
      <SignOutButton />
    </div>
  );
};

export const TopMenu: FC = async () => {
  const session = await auth();
  const isAdmin = await isGlobalAdmin();

  return (
    <div className="flex items-center justify-between border-b border-gray-200 bg-gray-50 py-2 pr-8 pl-72">
      <div className="flex items-center gap-4">
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
