"use client";

import { signIn } from "next-auth/react";
import { FC } from "react";

interface SignInButtonProps {
  provider?: string;
  children: React.ReactNode;
}

export const SignInButton: FC<SignInButtonProps> = ({ provider, children }) => {
  return (
    <button
      className="w-full rounded bg-blue-500 px-4 py-2 text-white hover:bg-blue-600"
      onClick={() => signIn(provider)}
    >
      {children}
    </button>
  );
};