import Link from "next/link";
import { listEnvs } from "../../server/api";

export default async function EnvsPage() {
  const envs = await listEnvs();

  return (
    <div className="p-4">
      <h1 className="mb-4 text-2xl font-bold">Environments</h1>
      {envs.length === 0 ? (
        <p>No environments found.</p>
      ) : (
        <ul className="space-y-2">
          {envs.map((env) => {
            // Remove file extension for cleaner URLs
            const envName = env.replace(/\.(yaml|yml)$/, "");
            return (
              <li key={env}>
                <Link
                  href={`/envs/view?name=${envName}`}
                  className="text-blue-600 hover:text-blue-800 hover:underline"
                >
                  {envName}
                </Link>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
