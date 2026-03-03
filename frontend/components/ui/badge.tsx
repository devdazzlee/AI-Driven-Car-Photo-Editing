import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium transition-colors",
  {
    variants: {
      variant: {
        default:
          "border-transparent bg-emerald-500/15 text-emerald-700 dark:text-emerald-400",
        secondary:
          "border-transparent bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300",
        destructive:
          "border-transparent bg-red-500/15 text-red-700 dark:text-red-400",
        outline: "border-slate-200 text-slate-700 dark:border-slate-700",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props} />
  );
}

export { Badge, badgeVariants };
