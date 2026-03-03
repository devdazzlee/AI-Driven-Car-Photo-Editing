"use client";

import { useState } from "react";
import { Download, X, ImageIcon, CheckCircle2 } from "lucide-react";
import { BeforeAfterSlider } from "./BeforeAfterSlider";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import type { ProcessedItem } from "@/types";

type Props = {
  results: ProcessedItem[];
  onClear: () => void;
  isLoading?: boolean;
};

export function ResultGallery({ results, onClear, isLoading }: Props) {
  const [selected, setSelected] = useState(0);
  const item = results[selected];

  if (isLoading) {
    return (
      <Card className="overflow-hidden border-slate-200/80 shadow-md dark:border-slate-700/80">
        <CardHeader className="space-y-2 pb-4 sm:pb-6">
          <div className="flex items-center gap-2">
            <Skeleton className="h-5 w-5 rounded" />
            <Skeleton className="h-5 w-32" />
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <Skeleton className="aspect-[4/3] w-full rounded-xl sm:aspect-[16/10]" />
          <div className="flex gap-2">
            <Skeleton className="h-9 w-24 rounded-lg" />
            <Skeleton className="h-9 w-32 rounded-lg" />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!item) return null;

  return (
    <Card
      className={cn(
        "animate-scale-in overflow-hidden border-slate-200/80 shadow-md dark:border-slate-700/80",
        "opacity-0 [animation-fill-mode:forwards]"
      )}
      style={{ animationDelay: "100ms" }}
    >
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4 sm:pb-6">
        <div className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-emerald-100 dark:bg-emerald-900/50">
            <CheckCircle2 className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
          </div>
          <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-100">
            Results
          </h2>
          <Badge variant="secondary" className="font-medium">
            {results.length} image{results.length > 1 ? "s" : ""}
          </Badge>
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={onClear}
          className="text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-100"
        >
          <X className="mr-1.5 h-4 w-4" />
          Clear all
        </Button>
      </CardHeader>

      <CardContent className="space-y-4 sm:space-y-6">
        <BeforeAfterSlider
          beforeSrc={item.originalUrl}
          afterSrc={item.processedUrl}
          beforeLabel="Original"
          afterLabel="Processed"
        />

        {results.length > 1 && (
          <div className="flex gap-2 overflow-x-auto pb-1 -mx-1 sm:mx-0">
            {results.map((r, i) => (
              <button
                key={i}
                onClick={() => setSelected(i)}
                className={cn(
                  "shrink-0 rounded-xl border px-3 py-2 text-xs font-medium transition-all duration-200 sm:px-4 sm:py-2.5",
                  selected === i
                    ? "border-emerald-500 bg-emerald-50 text-emerald-700 shadow-sm dark:border-emerald-600 dark:bg-emerald-950/50 dark:text-emerald-400"
                    : "border-slate-200 bg-white text-slate-600 hover:border-slate-300 hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-400 dark:hover:border-slate-600 dark:hover:bg-slate-800"
                )}
              >
                {r.originalFilename.length > 20
                  ? r.originalFilename.slice(0, 17) + "..."
                  : r.originalFilename}
              </button>
            ))}
          </div>
        )}

        <div className="flex flex-wrap gap-2">
          {results.map((r, i) => (
            <Button key={i} size="sm" asChild>
              <a
                href={r.processedUrl}
                download={r.processedFilename}
                target="_blank"
                rel="noopener noreferrer"
              >
                <Download className="mr-2 h-4 w-4" />
                Download {r.originalFilename.length > 25 ? "image" : r.originalFilename}
              </a>
            </Button>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
