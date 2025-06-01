import Link from "next/link";
import {
  getMettagridCfg,
  getMettagridCfgMap,
  MettagridCfg,
} from "../../../server/api";
import { FC } from "react";
import { z } from "zod";
import { ExtendedMapViewer } from "@/components/MapFileViewer";

interface EnvViewPageProps {
  searchParams: { path?: string };
}

const MapView: FC<{ cfg: MettagridCfg }> = async ({ cfg }) => {
  const map = await getMettagridCfgMap(cfg.path);
  return <ExtendedMapViewer mapData={map} />;
};

export default async function EnvViewPage({ searchParams }: EnvViewPageProps) {
  const cfgName = searchParams.path;

  if (!cfgName) {
    throw new Error("No config name provided");
  }

  const cfg = await getMettagridCfg(cfgName);

  return (
    <div className="p-4">
      <div className="mb-4">
        <Link
          href="/mettagrid-cfgs"
          className="text-blue-600 hover:text-blue-800 hover:underline"
        >
          ← Back to MettaGrid configs list
        </Link>
      </div>
      <h1 className="mb-4 text-2xl font-bold">
        Config: {cfg.path}, kind: {cfg.kind}
      </h1>
      <div className="rounded border bg-gray-50 p-4">
        <pre className="text-xs whitespace-pre-wrap">
          {JSON.stringify(cfg.cfg, null, 2)}
        </pre>
      </div>
      {(cfg.kind === "map" || cfg.kind === "env") && (
        <>
          <h1 className="mb-4 text-2xl font-bold">Generated Map</h1>
          <MapView cfg={cfg} />
        </>
      )}
    </div>
  );
}
