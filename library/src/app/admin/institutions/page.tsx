import { getAdminSessionOrRedirect } from "@/lib/adminAuth";
import { AdminInstitutionsView } from "@/components/AdminInstitutionsView";

/**
 * Admin Institutions Page
 *
 * Backend admin interface for managing institution owners
 * Restricted to global admins only
 */
export default async function AdminInstitutionsPage() {
  // Verify admin access (redirects if not admin)
  await getAdminSessionOrRedirect();

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="mx-auto max-w-7xl px-4 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">
            Institution Administration
          </h1>
          <p className="mt-2 text-gray-600">
            Manage institution owners and administrators
          </p>
        </div>

        <AdminInstitutionsView />
      </div>
    </div>
  );
}
