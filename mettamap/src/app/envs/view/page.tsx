import Link from "next/link";
import { getEnv } from "../../../server/api";

interface EnvViewPageProps {
  searchParams: { name?: string };
}

export default async function EnvViewPage({ searchParams }: EnvViewPageProps) {
  const envName = searchParams.name;

  if (!envName) {
    return (
      <div className="p-4">
        <h1 className="mb-4 text-2xl font-bold">Environment Viewer</h1>
        <p className="text-red-500">No environment name provided.</p>
        <Link href="/envs" className="text-blue-600 hover:text-blue-800 hover:underline">
          ← Back to environments list
        </Link>
      </div>
    );
  }

  try {
    const content = await getEnv(envName);

    return (
      <div className="p-4">
        <div className="mb-4">
          <Link href="/envs" className="text-blue-600 hover:text-blue-800 hover:underline">
            ← Back to environments list
          </Link>
        </div>
        <h1 className="mb-4 text-2xl font-bold">Environment: {envName}</h1>
        <div className="rounded border bg-gray-50 p-4">
          <pre className="whitespace-pre-wrap text-sm">{content}</pre>
        </div>
      </div>
    );
  } catch (error) {
    return (
      <div className="p-4">
        <div className="mb-4">
          <Link href="/envs" className="text-blue-600 hover:text-blue-800 hover:underline">
            ← Back to environments list
          </Link>
        </div>
        <h1 className="mb-4 text-2xl font-bold">Environment: {envName}</h1>
        <p className="text-red-500">
          Error loading environment: {error instanceof Error ? error.message : "Unknown error"}
        </p>
      </div>
    );
  }
} 
