"use client";

import { signOut } from "next-auth/react";
import { FC } from "react";

export const SignOutButton: FC = () => {
  return (
    <button
      className="cursor-pointer text-blue-500 hover:underline"
      onClick={() => signOut()}
    >
      Sign Out
    </button>
  );
};
