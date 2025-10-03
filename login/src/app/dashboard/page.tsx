import { getSessionOrRedirect } from "@/lib/auth";
import { SignOutButton } from "../components/SignOutButton";

export default async function Dashboard() {
  const session = await getSessionOrRedirect();

  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white shadow">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="flex h-16 justify-between">
            <div className="flex items-center">
              <h1 className="text-xl font-semibold">Login Service Dashboard</h1>
            </div>
            <div className="flex items-center">
              <span className="mr-4 text-sm text-gray-700">
                {session.user.email}
              </span>
              <SignOutButton />
            </div>
          </div>
        </div>
      </nav>

      <main className="mx-auto max-w-7xl py-6 sm:px-6 lg:px-8">
        <div className="px-4 py-6 sm:px-0">
          <div className="rounded-lg border-4 border-dashed border-gray-200 p-8">
            <div className="text-center">
              <h2 className="text-2xl font-bold text-gray-900">
                Welcome, {session.user.name || session.user.email}!
              </h2>
              <p className="mt-2 text-gray-600">
                You have successfully authenticated with the login service.
              </p>

              <div className="mt-6 rounded-lg bg-blue-50 p-4">
                <h3 className="text-lg font-medium text-blue-900">
                  User Details
                </h3>
                <div className="mt-2 text-left">
                  <p className="text-sm text-blue-800">
                    <strong>ID:</strong> {session.user.id}
                  </p>
                  <p className="text-sm text-blue-800">
                    <strong>Email:</strong> {session.user.email}
                  </p>
                  {session.user.name && (
                    <p className="text-sm text-blue-800">
                      <strong>Name:</strong> {session.user.name}
                    </p>
                  )}
                </div>
              </div>

              <div className="mt-6">
                <h3 className="text-lg font-medium text-gray-900">
                  Available API Endpoints
                </h3>
                <div className="mt-2 space-y-1 text-sm text-gray-600">
                  <p>
                    <code className="rounded bg-gray-100 px-1">
                      GET /api/user
                    </code>{" "}
                    - Get current user information
                  </p>
                  <p>
                    <code className="rounded bg-gray-100 px-1">
                      GET /api/validate
                    </code>{" "}
                    - Validate current session
                  </p>
                  <p>
                    <code className="rounded bg-gray-100 px-1">
                      POST /api/validate
                    </code>{" "}
                    - Validate session token
                  </p>
                  <p>
                    <code className="rounded bg-gray-100 px-1">
                      GET /api/health
                    </code>{" "}
                    - Service health check
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}