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
    <div className="mx-auto max-w-7xl px-4 py-4 md:py-8">
      <div className="mb-6 md:mb-8">
        <h1 className="text-2xl font-bold text-gray-900 md:text-3xl">
          User Administration
        </h1>
        <p className="mt-2 text-sm text-gray-600 md:text-base">
          Manage users and moderation actions
        </p>
      </div>

      <AdminUsersView />
    </div>
  );
}
