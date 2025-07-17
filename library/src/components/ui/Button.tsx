import { clsx } from "clsx";
import { FC, PropsWithChildren } from "react";

type ButtonTheme = "default" | "primary";

export type ButtonProps = PropsWithChildren<{
  onClick?: () => void;
  theme?: ButtonTheme;
  disabled?: boolean;
  size?: "small" | "medium";
  // We default to type="button", to avoid form-related bugs.
  // In HTML standard, there's also "reset", but it's rarely useful.
  type?: "submit" | "button";
}>;

// For internal use only, for now (see ButtonWithDropdown).
export const ButtonGroup: FC<PropsWithChildren> = ({ children }) => {
  return <div className="button-group flex items-center">{children}</div>;
};

export const Button: FC<ButtonProps> = ({
  onClick,
  theme = "default",
  disabled,
  size = "medium",
  type = "button",
  children,
}) => {
  return (
    <button
      className={clsx(
        "border text-sm font-medium",
        theme === "primary" && "border-green-900 bg-green-700 text-white",
        theme === "default" && "border-slate-300 bg-slate-100 text-gray-600",

        disabled
          ? "opacity-60"
          : [
              theme === "primary" &&
                "hover:border-green-800 hover:bg-green-800",
              theme === "default" && "hover:bg-slate-200 hover:text-gray-900",
            ],

        size === "medium" && "h-8 rounded-md",
        size === "small" && "h-6 rounded-sm",
        // This could probably be simplified, but I'm not sure how.
        // Tailwind group-* styles don't allow styling based on parent, only on parent state.
        "[.button-group_&:not(:first-child)]:rounded-l-none",
        "[.button-group_&:not(:first-child)]:border-l-0",
        "[.button-group_&:not(:last-child)]:rounded-r-none"
      )}
      onClick={onClick}
      disabled={disabled}
      type={type}
    >
      <div
        className={clsx(
          "flex items-center justify-center space-x-1",
          size === "medium" && "px-4",
          size === "small" && "px-3"
        )}
      >
        {children}
      </div>
    </button>
  );
};
