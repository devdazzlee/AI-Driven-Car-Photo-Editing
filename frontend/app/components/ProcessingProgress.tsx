"use client";

import { Loader2 } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { Card, CardContent } from "@/components/ui/card";

type Props = {
  completed: number;
  total: number;
};

export function ProcessingProgress({ completed, total }: Props) {
  const pct = total > 0 ? Math.round((completed / total) * 100) : 0;

  return (
    <Card className="animate-scale-in border-emerald-200/60 bg-emerald-50/30 dark:border-emerald-900/40 dark:bg-emerald-950/20">
      <CardContent className="p-5 sm:p-6">
        <div className="flex items-center justify-between gap-4">
          <span className="flex items-center gap-2 text-sm font-semibold text-slate-700 dark:text-slate-200">
            <Loader2 className="h-4 w-4 animate-spin text-emerald-600" />
            Processing… {completed} of {total} done
          </span>
          <span className="text-sm font-medium text-emerald-600 dark:text-emerald-400">
            {pct}%
          </span>
        </div>
        <Progress value={pct} className="mt-3" />
      </CardContent>
    </Card>
  );
}
