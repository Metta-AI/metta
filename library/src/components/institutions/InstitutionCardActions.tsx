"use client";

import React from "react";

import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

type ActionState = "idle" | "loading" | "disabled";

interface InstitutionActionButtonProps {
  state?: ActionState;
  onClick?: () => void;
  children: React.ReactNode;
}

export const InstitutionActionButton: React.FC<
  InstitutionActionButtonProps
> = ({ state = "idle", onClick, children }) => {
  const isDisabled = state === "disabled" || state === "loading";

  return (
    <Button size="sm" onClick={onClick} disabled={isDisabled}>
      {state === "loading" ? "Working..." : children}
    </Button>
  );
};

interface InstitutionStatusBadgeProps {
  label: string;
  tone?: "neutral" | "info" | "warning" | "danger";
}

export const InstitutionStatusBadge: React.FC<InstitutionStatusBadgeProps> = ({
  label,
  tone = "neutral",
}) => {
  const toneClass = {
    neutral: "bg-slate-100 text-slate-700",
    info: "bg-blue-100 text-blue-700",
    warning: "bg-amber-100 text-amber-800",
    danger: "bg-rose-100 text-rose-700",
  }[tone];

  return (
    <Badge className={toneClass} variant="secondary">
      {label}
    </Badge>
  );
};
