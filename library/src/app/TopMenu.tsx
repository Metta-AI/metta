import { auth, signOut } from "@/lib/auth";
import Link from "next/link";
import { FC } from "react";
import { SignOutButton } from "./SignOutButton";

const SignInButton: FC = () => {
  return (
    <Link href="/api/auth/signin" className="text-blue-500 hover:underline">
      Sign In
    </Link>
  );
};

const UserInfo: FC = async () => {
  const session = await auth();

  if (session?.user?.email) {
    return (
      <div className="flex items-center gap-2">
        <span>{session.user.email}</span>
        <SignOutButton />
      </div>
    );
  }

  return <SignInButton />;
};

export const TopMenu: FC = async () => {
  const session = await auth();

  return (
    <div className="flex items-center justify-between border-b border-gray-200 bg-gray-50 px-8 py-2">
      <div className="flex items-center gap-4">
        <Link href="/" className="font-bold">
          Softmax Library
        </Link>
        {/* Add more links here */}
      </div>
      <UserInfo />
    </div>
  );
};
