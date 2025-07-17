import "./globals.css";

import type { Metadata } from "next";
import { SessionProvider } from "next-auth/react";
import { redirect } from "next/navigation";
import { NuqsAdapter } from "nuqs/adapters/next/app";
import { FC, PropsWithChildren } from "react";

import { LibraryLayout } from "@/components/LibraryLayout";
import { auth } from "@/lib/auth";

export const metadata: Metadata = {
  title: "Softmax Library",
};

const GlobalProviders: FC<PropsWithChildren> = async ({ children }) => {
  // Configure any other global providers here
  return (
    <SessionProvider>
      <NuqsAdapter>{children}</NuqsAdapter>
    </SessionProvider>
  );
};

export default async function RootLayout({ children }: PropsWithChildren) {
  const session = await auth();
  if (!session) {
    redirect("/api/auth/signin");
  }

  return (
    <html lang="en">
      <body className="overflow-y-scroll">
        <GlobalProviders>
          <LibraryLayout>{children}</LibraryLayout>
        </GlobalProviders>
      </body>
    </html>
  );
}

export const dynamic = "force-dynamic";
