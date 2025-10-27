"use client";

import React from "react";

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { cn } from "@/lib/utils";

type OverlaySize = "md" | "lg" | "xl" | "full";

const sizeClassName: Record<OverlaySize, string> = {
  md: "sm:max-w-xl",
  lg: "sm:max-w-2xl",
  xl: "sm:max-w-4xl",
  full: "h-[90vh] w-[90vw] sm:max-w-6xl",
};

export interface BaseOverlayProps {
  open: boolean;
  title?: React.ReactNode;
  description?: React.ReactNode;
  onClose: () => void;
  children: React.ReactNode;
  footer?: React.ReactNode;
  size?: OverlaySize;
  dismissible?: boolean;
  className?: string;
  contentClassName?: string;
}

export const BaseOverlay: React.FC<BaseOverlayProps> = ({
  open,
  title,
  description,
  onClose,
  children,
  footer,
  size = "xl",
  dismissible = true,
  className,
  contentClassName,
}) => {
  return (
    <Dialog open={open} onOpenChange={(nextOpen) => !nextOpen && onClose()}>
      <DialogContent
        showCloseButton={dismissible}
        className={cn(sizeClassName[size], "max-h-[90vh]", className)}
      >
        {(title || description) && (
          <DialogHeader className="gap-1 text-left">
            {title &&
              (typeof title === "string" ? (
                <DialogTitle>{title}</DialogTitle>
              ) : (
                title
              ))}
            {description &&
              (typeof description === "string" ? (
                <DialogDescription>{description}</DialogDescription>
              ) : (
                description
              ))}
          </DialogHeader>
        )}

        <div
          className={cn(
            "scrollbar-thin max-h-[calc(90vh-12rem)] overflow-y-auto",
            contentClassName
          )}
        >
          {children}
        </div>

        {footer && <DialogFooter>{footer}</DialogFooter>}
      </DialogContent>
    </Dialog>
  );
};
