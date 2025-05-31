import { PropsWithChildren } from "react";

import { MapFilter } from "./MapFilter";

export default function MapsLayout({ children }: PropsWithChildren) {
  return (
    <div>
      <MapFilter />
      <div className="p-8">{children}</div>
    </div>
  );
}
