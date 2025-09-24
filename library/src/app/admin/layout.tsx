import Link from "next/link";
import { Users, Building2 } from "lucide-react";

import { getAdminSessionOrRedirect } from "@/lib/adminAuth";

/**
 * Admin Layout
 *
 * Ensures all admin routes are protected and only accessible to global admins
 */
export default async function AdminLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  // This will redirect if user is not an admin
  await getAdminSessionOrRedirect();

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="border-b border-red-200 bg-red-50 p-2 text-center">
        <span className="text-sm font-medium text-red-800">
          🔒 Administration Panel - Authorized Personnel Only
        </span>
      </div>

      {/* Admin Navigation */}
      <div className="border-b border-gray-200 bg-white">
        <div className="mx-auto max-w-7xl px-4">
          <nav className="flex space-x-8 py-4">
            <Link
              href="/admin/users"
              className="flex items-center gap-2 rounded-md px-3 py-2 text-sm font-medium text-gray-700 transition-colors hover:bg-gray-100 hover:text-gray-900"
            >
              <Users className="h-4 w-4" />
              User Management
            </Link>
            <Link
              href="/admin/institutions"
              className="flex items-center gap-2 rounded-md px-3 py-2 text-sm font-medium text-gray-700 transition-colors hover:bg-gray-100 hover:text-gray-900"
            >
              <Building2 className="h-4 w-4" />
              Institutions
            </Link>
          </nav>
        </div>
      </div>

      {children}
    </div>
  );
}
