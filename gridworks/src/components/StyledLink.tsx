import clsx from "clsx";
import Link from "next/link";

export const StyledLink: React.FC<
  React.AnchorHTMLAttributes<HTMLAnchorElement> & { href: string }
> = ({ className, ...props }) => (
  <Link
    {...props}
    className={clsx(
      className,
      "text-blue-600 hover:text-blue-800 hover:underline"
    )}
  />
);
