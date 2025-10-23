"use client";

import React from "react";

export const OverlayLoadingSkeleton: React.FC = () => (
  <div className="rounded-lg bg-white p-6 shadow-xl">
    <div className="animate-pulse">
      <div className="mb-4 h-8 rounded bg-gray-200" />
      <div className="space-y-3">
        <div className="h-4 w-3/4 rounded bg-gray-200" />
        <div className="h-4 w-1/2 rounded bg-gray-200" />
        <div className="h-4 w-5/6 rounded bg-gray-200" />
      </div>
    </div>
  </div>
);
