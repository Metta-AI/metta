import Link from "next/link";
import { Users, Building2, Shield, AlertTriangle } from "lucide-react";

import { getAdminSessionOrRedirect } from "@/lib/adminAuth";

/**
 * Admin Dashboard
 *
 * Main landing page for admin features
 */
export default async function AdminPage() {
  // Verify admin access (redirects if not admin)
  await getAdminSessionOrRedirect();

  return (
    <div className="flex h-full w-full flex-col">
      {/* Header Section */}
      <div className="border-b border-gray-200 bg-white px-4 py-4 md:px-6 md:py-6">
        <div className="flex flex-col gap-2">
          <h1 className="text-xl font-semibold text-gray-900">
            Administration Dashboard
          </h1>
          <p className="text-sm text-gray-600">
            Manage users, institutions, and system-wide settings
          </p>
        </div>
      </div>

      {/* Content Area */}
      <div className="flex-1 overflow-y-auto px-4 py-6 md:px-6">
        <div className="mx-auto w-full max-w-7xl space-y-6">
          {/* Warning Banner */}
          <div className="rounded-lg border border-amber-200 bg-amber-50 p-4">
            <div className="flex items-start gap-3">
              <AlertTriangle className="mt-0.5 h-5 w-5 flex-shrink-0 text-amber-600" />
              <div className="text-sm">
                <h3 className="font-medium text-amber-800">Admin Access</h3>
                <p className="mt-1 text-amber-700">
                  You have global administrator privileges. Use these tools
                  carefully as they affect all users and data.
                </p>
              </div>
            </div>
          </div>

          {/* Admin Sections */}
          <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
            {/* User Management */}
            <Link
              href="/admin/users"
              className="group rounded-lg border border-gray-200 bg-white p-6 shadow-sm transition-all hover:border-gray-300 hover:shadow-md"
            >
              <div className="flex items-center gap-4">
                <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-blue-600 text-white">
                  <Users className="h-6 w-6" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 group-hover:text-blue-600">
                    User Management
                  </h3>
                  <p className="mt-1 text-sm text-gray-600">
                    View, search, ban, and unban users across the platform
                  </p>
                </div>
              </div>
              <div className="mt-4 flex items-center text-sm text-blue-600 group-hover:text-blue-700">
                <span>Manage users</span>
                <svg
                  className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 5l7 7-7 7"
                  />
                </svg>
              </div>
            </Link>

            {/* Institution Management */}
            <Link
              href="/admin/institutions"
              className="group rounded-lg border border-gray-200 bg-white p-6 shadow-sm transition-all hover:border-gray-300 hover:shadow-md"
            >
              <div className="flex items-center gap-4">
                <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-green-600 text-white">
                  <Building2 className="h-6 w-6" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 group-hover:text-green-600">
                    Institution Administration
                  </h3>
                  <p className="mt-1 text-sm text-gray-600">
                    Manage institution admins, approval settings, and membership
                    requests
                  </p>
                </div>
              </div>
              <div className="mt-4 flex items-center text-sm text-green-600 group-hover:text-green-700">
                <span>Manage institutions</span>
                <svg
                  className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 5l7 7-7 7"
                  />
                </svg>
              </div>
            </Link>
          </div>

          {/* System Info */}
          <div className="rounded-lg border border-gray-200 bg-white p-6">
            <div className="mb-4 flex items-center gap-2">
              <Shield className="h-5 w-5 text-gray-600" />
              <h3 className="text-lg font-semibold text-gray-900">
                System Information
              </h3>
            </div>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <div>
                <dt className="text-sm font-medium text-gray-600">
                  Admin Level
                </dt>
                <dd className="text-lg font-semibold text-gray-900">
                  Global Administrator
                </dd>
              </div>
              <div>
                <dt className="text-sm font-medium text-gray-600">
                  Access Scope
                </dt>
                <dd className="text-lg font-semibold text-gray-900">
                  All Users & Institutions
                </dd>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
