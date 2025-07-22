import { FC, PropsWithChildren } from "react";

// Add more props here or create multiple CustomPageLayout components to represent different page layouts.
export const PageLayout: FC<PropsWithChildren> = ({ children }) => {
  return <div className="p-4">{children}</div>;
};
