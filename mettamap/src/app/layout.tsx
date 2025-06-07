import "./globals.css";

import type { Metadata } from "next";
import Link from "next/link";
import { NuqsAdapter } from "nuqs/adapters/next/app";
import { FC, PropsWithChildren, Suspense } from "react";

import { RepoRootProvider } from "@/components/RepoRootContext";
import { getRepoRoot } from "@/lib/api";

export const metadata: Metadata = {
  title: "MettaMap viewer",
};

const GlobalProviders: FC<PropsWithChildren> = async ({ children }) => {
  const repoRoot = await getRepoRoot();
  return (
    <RepoRootProvider root={repoRoot}>
      <Suspense>
        <NuqsAdapter>{children}</NuqsAdapter>
      </Suspense>
    </RepoRootProvider>
  );
};

const TopMenu: FC = () => {
  return (
    <div className="flex items-center gap-4 border-b border-gray-200 bg-gray-100 px-8 py-2">
      <Link href="/" className="font-bold">
        MettaMap Viewer
      </Link>
      <Link
        href="/mettagrid-cfgs"
        className="text-sm font-medium text-gray-600 hover:text-gray-900"
      >
        MettaGrid Configs
      </Link>
      <Link
        href="/stored-maps"
        className="text-sm font-medium text-gray-600 hover:text-gray-900"
      >
        Stored Maps
      </Link>
    </div>
  );
};

export default function RootLayout({ children }: PropsWithChildren) {
  return (
    <html lang="en">
      <body className="overflow-y-scroll">
        <GlobalProviders>
          <div className="flex h-screen flex-col">
            <TopMenu />
            {children}
          </div>
        </GlobalProviders>
      </body>
    </html>
  );
}
