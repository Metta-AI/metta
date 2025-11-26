import { listConfigMakers } from "@/lib/api";
import { ConfigRenderer } from "./ConfigRenderer";

export default async function EnvsPage() {
  const cfgs = await listConfigMakers();

  return (
    <div className="p-4">
      <h1 className="mb-4 text-2xl font-bold">Config Makers</h1>
      <ConfigRenderer initialCfgs={cfgs} />
    </div>
  );
}

export const dynamic = "force-dynamic";
