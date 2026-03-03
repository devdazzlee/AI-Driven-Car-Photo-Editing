"use client";

import { useId } from "react";
import { cn } from "@/lib/utils";

/**
 * AI-style icon: neural network design for Car Image AI
 */
export function AIIcon({ className, size = 36 }: { className?: string; size?: number }) {
  const id = useId().replace(/:/g, "-");
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 36 36"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={cn("shrink-0", className)}
      aria-hidden
    >
      <circle cx="18" cy="18" r="15" stroke="currentColor" strokeWidth="2" strokeOpacity="0.3" fill="none" />
      <circle cx="18" cy="10" r="2.5" fill="currentColor" />
      <circle cx="12" cy="16" r="2.5" fill="currentColor" />
      <circle cx="24" cy="16" r="2.5" fill="currentColor" />
      <circle cx="18" cy="26" r="2.5" fill="currentColor" />
      <circle cx="12" cy="22" r="1.5" fill="currentColor" fillOpacity="0.7" />
      <circle cx="24" cy="22" r="1.5" fill="currentColor" fillOpacity="0.7" />
      <path
        d="M18 12.5 L12 13.5 M18 12.5 L24 13.5 M12 18.5 L18 23.5 M24 18.5 L18 23.5 M12 18.5 L12 20 M24 18.5 L24 20"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeOpacity="0.6"
      />
      <circle cx="18" cy="18" r="4" fill={`url(#ai-gradient-${id})`} opacity="0.9" />
      <defs>
        <linearGradient id={`ai-gradient-${id}`} x1="14" y1="14" x2="22" y2="22" gradientUnits="userSpaceOnUse">
          <stop stopColor="currentColor" stopOpacity="1" />
          <stop offset="1" stopColor="currentColor" stopOpacity="0.6" />
        </linearGradient>
      </defs>
    </svg>
  );
}
