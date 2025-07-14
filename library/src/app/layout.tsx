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
		<div className="flex items-center justify-between border-b border-gray-200 bg-gray-50 px-8 py-2">
			<div className="flex items-center gap-4">
				<Link href="/" className="font-bold">
					Softmax Library
				</Link>
				{/* Add more links here */}
			</div>
			<div className="flex items-center gap-4">
				{/* <SignedOut>
					<SignInButton />
					<SignUpButton>
						<button className="bg-[#6c47ff] text-white rounded-full font-medium text-sm sm:text-base h-10 sm:h-12 px-4 sm:px-5 cursor-pointer">
							Sign Up
						</button>
					</SignUpButton>
				</SignedOut>
				<SignedIn>
					<UserButton />
				</SignedIn> */}
			</div>
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
