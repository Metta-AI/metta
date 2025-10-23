import { getAdminSessionOrRedirect } from "@/lib/adminAuth";
import { AdminUsersView } from "@/components/AdminUsersView";

/**
 * Admin Users Page
 *
 * Backend admin interface for managing users (ban/unban)
 * Restricted to global admins only
 */
export default async function AdminUsersPage() {
  // Verify admin access (redirects if not admin)
  await getAdminSessionOrRedirect();

  return (
    <div className="flex h-full w-full flex-col">
      {/* Header Section */}
      <div className="border-b border-gray-200 bg-white px-4 py-4 md:px-6 md:py-6">
        <div className="flex flex-col gap-2">
          <h1 className="text-xl font-semibold text-gray-900">
            User Administration
          </h1>
          <p className="text-sm text-gray-600">
            Manage users and moderation actions
          </p>
        </div>
      </div>

      {/* Content Area */}
      <div className="flex-1 overflow-y-auto px-4 py-6 md:px-6">
        <div className="mx-auto w-full max-w-7xl">
          <AdminUsersView />
        </div>
      </div>
    </div>
  );
}
