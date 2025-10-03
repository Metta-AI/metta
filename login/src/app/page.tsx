import { auth } from "@/lib/auth";
import { EmailSignInForm } from "./components/EmailSignInForm";
import { SignInButton } from "./components/SignInButton";
import { SignOutButton } from "./components/SignOutButton";

export default async function Home() {
  const session = await auth();

  if (session?.user) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-gray-50">
        <div className="max-w-md rounded-lg bg-white p-8 shadow-md">
          <h1 className="mb-4 text-2xl font-bold text-gray-900">
            Welcome to Login Service
          </h1>
          <div className="mb-4 rounded-lg bg-green-50 p-4">
            <p className="text-sm text-green-800">
              ✅ Successfully authenticated as{" "}
              <strong>{session.user.email}</strong>
            </p>
            {session.user.name && (
              <p className="text-sm text-green-800">
                Name: <strong>{session.user.name}</strong>
              </p>
            )}
          </div>
          <div className="flex justify-center">
            <SignOutButton />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-50">
      <div className="max-w-md rounded-lg bg-white p-8 shadow-md">
        <h1 className="mb-4 text-2xl font-bold text-gray-900">
          Login Service
        </h1>
        <p className="mb-6 text-gray-600">
          Please sign in to access Softmax applications
        </p>
        <div className="space-y-4">
          {process.env.DEV_MODE === "true" && (
            <div>
              <EmailSignInForm />
            </div>
          )}
          <div className="relative">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-gray-300" />
            </div>
            <div className="relative flex justify-center text-sm">
              <span className="bg-white px-2 text-gray-500">Or</span>
            </div>
          </div>
          <div>
            <SignInButton provider="google">Sign in with Google</SignInButton>
          </div>
        </div>
      </div>
    </div>
  );
}