import { auth } from "@/lib/auth";

export default async function Home() {
  const session = await auth();
  console.log({ session });

  return <div>Hello {session?.user?.email ?? "No user"}</div>;
}
