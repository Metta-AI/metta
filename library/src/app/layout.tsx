import "./globals.css";

import { FC, PropsWithChildren } from "react";

import type { Metadata } from "next";
import { NuqsAdapter } from "nuqs/adapters/next/app";
import { PageLayout } from "@/components/ui/PageLayout";
import { SessionProvider } from "next-auth/react";
import { TopMenu } from "./TopMenu";

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

export default function RootLayout({ children }: PropsWithChildren) {
  return (
    <html lang="en">
      <body className="overflow-y-scroll">
        <GlobalProviders>
          <div className="flex h-screen flex-col">
            <TopMenu />
            <PageLayout>{children}</PageLayout>
          </div>
        </GlobalProviders>
      </body>
    </html>
  );
}
