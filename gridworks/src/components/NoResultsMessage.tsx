import { FC } from "react";
import clsx from "clsx";

export const NoResultsMessage: FC<{
  show: boolean;
  message?: string;
  className?: string;
}> = ({ show, message = "No Results Found", className = "mb-4" }) => {
  return (
    <div className={clsx(!show && "hidden", className)}>
      <p className="text-gray-500">{message}</p>
    </div>
  );
};
