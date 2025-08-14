import React from "react";
import { createRoot } from "react-dom/client";
import { EvalFinder } from "./EvalFinder";

function render({ model, el }: { model: any; el: HTMLElement }) {
  const root = createRoot(el);
  root.render(<EvalFinder model={model} />);
  return () => root.unmount();
}

export default { render };
