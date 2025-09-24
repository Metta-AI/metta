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
      {children}
    </div>
  );
}
