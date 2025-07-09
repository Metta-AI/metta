import "./globals.css";

import type { Metadata } from "next";
import Link from "next/link";
import { NuqsAdapter } from "nuqs/adapters/next/app";
import { FC, PropsWithChildren, Suspense } from "react";

import { RepoRootProvider } from "@/components/RepoRootContext";
import { getRepoRoot } from "@/lib/api";

import { TopMenuLink } from "./TopMenuLink";

export const metadata: Metadata = {
  title: "Gridworks",
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
    <div className="flex items-center gap-6 border-b border-gray-200 bg-gray-50 px-8 py-2">
      <Link href="/" className="font-bold">
        Gridworks
      </Link>
      <TopMenuLink href="/mettagrid-cfgs">MettaGrid Configs</TopMenuLink>
      <TopMenuLink href="/map-editor">Map Editor</TopMenuLink>
      <TopMenuLink href="/stored-maps">Stored Maps (Experimental)</TopMenuLink>
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

// Opt out of all static rendering, this is a local app so it doesn't matter.
export const dynamic = "force-dynamic";
