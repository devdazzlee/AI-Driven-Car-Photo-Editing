import { cn } from "@/lib/utils";

function Skeleton({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "rounded-lg bg-slate-200 dark:bg-slate-800",
        "animate-pulse",
        className
      )}
      {...props}
    />
  );
}

export { Skeleton };
