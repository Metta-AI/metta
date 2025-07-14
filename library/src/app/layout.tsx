import "./globals.css";

import { FC, PropsWithChildren, Suspense } from "react";

import Link from "next/link";
import type { Metadata } from "next";
import { NuqsAdapter } from "nuqs/adapters/next/app";
import { PageLayout } from "@/components/ui/PageLayout";

export const metadata: Metadata = {
	title: "Softmax Library",
};

const GlobalProviders: FC<PropsWithChildren> = async ({ children }) => {
	// Configure any other global providers here
	return (
		<Suspense>
			<NuqsAdapter>{children}</NuqsAdapter>
		</Suspense>
	);
};

const TopMenu: FC = () => {
	return (
		<div className="flex items-center gap-6 border-b border-gray-200 bg-gray-50 px-8 py-2">
			<Link href="/" className="font-bold">
				Softmax Library
			</Link>
			{/* Add more links here */}
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
						<PageLayout>{children}</PageLayout>
					</div>
				</GlobalProviders>
			</body>
		</html>
	);
}
