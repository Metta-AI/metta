import "./globals.css";

import type { Metadata } from "next";
import { SessionProvider } from "next-auth/react";
import { redirect } from "next/navigation";
import { NuqsAdapter } from "nuqs/adapters/next/app";
import { FC, PropsWithChildren } from "react";
import { Toaster } from "sonner";

import { LibraryLayout } from "@/components/LibraryLayout";
import { MathJaxProvider } from "@/components/MathJaxProvider";
import { QueryProvider } from "@/lib/query-client";
import { auth } from "@/lib/auth";

export const metadata: Metadata = {
  title: "Softmax Library",
};

const GlobalProviders: FC<PropsWithChildren> = async ({ children }) => {
  // Configure any other global providers here
  return (
    <SessionProvider>
      <QueryProvider>
        <NuqsAdapter>
          <MathJaxProvider>{children}</MathJaxProvider>
        </NuqsAdapter>
      </QueryProvider>
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
      <body className="overflow-y-scroll" suppressHydrationWarning={true}>
        <GlobalProviders>
          <LibraryLayout>{children}</LibraryLayout>
          <Toaster position="top-right" richColors closeButton />
        </GlobalProviders>
      </body>
    </html>
  );
}

export const dynamic = "force-dynamic";
