import { redirect } from "next/navigation";
import { auth, isSignedIn } from "@/lib/auth";
import { SettingsView } from "@/components/SettingsView";

export default async function SettingsPage() {
  const session = await auth();

  if (!isSignedIn(session)) {
    redirect("/api/auth/signin");
  }

  return <SettingsView user={session.user} />;
}

