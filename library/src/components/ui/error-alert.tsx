import { AlertCircle, XCircle } from "lucide-react";
import { cn } from "@/lib/utils";

interface ErrorAlertProps {
  title?: string;
  message: string;
  variant?: "inline" | "block";
  onRetry?: () => void;
  onDismiss?: () => void;
  className?: string;
}

export function ErrorAlert({
  title = "Error",
  message,
  variant = "inline",
  onRetry,
  onDismiss,
  className,
}: ErrorAlertProps) {
  if (variant === "inline") {
    return (
      <div
        className={cn(
          "rounded-lg border border-red-200 bg-red-50 p-3",
          className
        )}
        role="alert"
      >
        <div className="flex">
          <div className="flex-shrink-0">
            <AlertCircle className="h-5 w-5 text-red-400" />
          </div>
          <div className="ml-3 flex-1">
            <p className="text-sm text-red-700">{message}</p>
          </div>
          {onDismiss && (
            <button
              onClick={onDismiss}
              className="ml-auto flex-shrink-0 text-red-400 hover:text-red-600"
              aria-label="Dismiss"
            >
              <XCircle className="h-5 w-5" />
            </button>
          )}
        </div>
        {onRetry && (
          <div className="mt-2">
            <button
              onClick={onRetry}
              className="text-sm font-medium text-red-600 hover:text-red-800 hover:underline"
            >
              Try again
            </button>
          </div>
        )}
      </div>
    );
  }

  return (
    <div
      className={cn(
        "flex min-h-[200px] flex-col items-center justify-center gap-3 rounded-lg border border-red-200 bg-red-50 p-6",
        className
      )}
      role="alert"
    >
      <AlertCircle className="h-12 w-12 text-red-400" />
      <div className="text-center">
        <h3 className="text-lg font-semibold text-red-900">{title}</h3>
        <p className="mt-1 text-sm text-red-700">{message}</p>
      </div>
      {onRetry && (
        <button
          onClick={onRetry}
          className="mt-2 rounded-md bg-red-600 px-4 py-2 text-sm font-medium text-white hover:bg-red-700"
        >
          Retry
        </button>
      )}
    </div>
  );
}
