import { FC } from "react";
import { FallbackProps } from "react-error-boundary";

export const ErrorFallback: FC<FallbackProps> = ({ error }) => {
  return (
    <div className="rounded-md bg-red-100 p-4">
      <header className="text-bold mb-4 text-xl text-red-700">Error</header>
      <pre>{error.message}</pre>
    </div>
  );
};
