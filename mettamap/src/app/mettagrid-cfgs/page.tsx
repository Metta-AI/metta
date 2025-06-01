import Link from "next/link";
import { listMettagridCfgs } from "../../server/api";

export default async function EnvsPage() {
  const cfgs = await listMettagridCfgs();

  return (
    <div className="p-4">
      <h1 className="mb-4 text-2xl font-bold">MettaGrid Configs</h1>
      {Object.keys(cfgs).map((kind) => {
        return (
          <div key={kind}>
            <h2 className="text-lg font-bold">{kind}</h2>
            <ul className="space-y-2">
              {cfgs[kind as keyof typeof cfgs].map((cfg) => {
                return (
                  <li key={cfg.path}>
                    <Link
                      href={`/mettagrid-cfgs/view?path=${cfg.path}`}
                      className="text-blue-600 hover:text-blue-800 hover:underline"
                    >
                      {cfg.path}
                    </Link>
                  </li>
                );
              })}
            </ul>
          </div>
        );
      })}
    </div>
  );
}
