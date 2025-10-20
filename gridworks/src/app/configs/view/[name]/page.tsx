import { StyledLink } from "@/components/StyledLink";
import { configsRoute } from "@/lib/routes";

import { getConfig } from "../../../../lib/api";
import { ConfigViewScreen } from "./ConfigViewScreen";

interface ConfigViewPage {
  params: Promise<{ name: string }>;
}

export default async function ConfigViewPage({ params }: ConfigViewPage) {
  const { name } = await params;

  if (!name) {
    throw new Error("No config name provided");
  }

  const cfg = await getConfig(name);

  return (
    <div>
      <div className="border-b border-gray-200 bg-gray-100 p-4 pb-6">
        <div className="mb-4">
          <StyledLink href={configsRoute()}>
            ← Back to config makers list
          </StyledLink>
        </div>
        <h1 className="text-2xl font-bold">
          <span>
            <a
              className="hover:underline"
              href={`cursor://file${cfg.maker.absolute_path}:${cfg.maker.line}`}
            >
              {cfg.maker.path}
            </a>
          </span>
          <span className="text-xl text-gray-400"> &rarr;</span>
          <span className="text-xl text-gray-500"> {cfg.maker.kind}</span>
        </h1>
      </div>
      <div className="p-4">
        <ConfigViewScreen cfg={cfg} />
      </div>
    </div>
  );
}
