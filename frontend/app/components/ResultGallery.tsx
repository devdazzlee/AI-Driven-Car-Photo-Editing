"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Download, X, CheckCircle2, ChevronLeft, ChevronRight, DownloadCloud, Loader2 } from "lucide-react";
import { toast } from "sonner";
import JSZip from "jszip";
import { BeforeAfterSlider } from "./BeforeAfterSlider";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import type { ProcessedItem } from "@/types";

type Props = {
  results: ProcessedItem[];
  onClear: () => void;
  isLoading?: boolean;
};

function truncateFilename(name: string, maxLen = 22) {
  if (name.length <= maxLen) return name;
  const ext = name.slice(name.lastIndexOf("."));
  const base = name.slice(0, name.lastIndexOf("."));
  if (base.length <= 6) return name;
  return base.slice(0, maxLen - ext.length - 3) + "…" + ext;
}

export function ResultGallery({ results, onClear, isLoading }: Props) {
  const [selected, setSelected] = useState(0);
  const [downloadAllLoading, setDownloadAllLoading] = useState(false);
  const downloadAllRef = useRef(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const item = results[selected];

  useEffect(() => {
    const el = scrollRef.current;
    if (!el || results.length <= 1) return;
    const tab = el.querySelector(`[data-index="${selected}"]`);
    tab?.scrollIntoView({ behavior: "instant", block: "nearest", inline: "center" });
  }, [selected, results.length]);

  const handleDownloadAll = useCallback(async () => {
    if (downloadAllRef.current || results.length === 0) return;
    downloadAllRef.current = true;
    setDownloadAllLoading(true);
    try {
      const zip = new JSZip();
      for (let i = 0; i < results.length; i++) {
        const r = results[i];
        const res = await fetch(r.processedUrl);
        const blob = await res.blob();
        zip.file(r.processedFilename, blob);
      }
      const zipBlob = await zip.generateAsync({ type: "blob" });
      const url = URL.createObjectURL(zipBlob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `car-images-processed.zip`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      toast.success(`Downloaded ${results.length} image${results.length > 1 ? "s" : ""} as ZIP`);
    } catch (e) {
      toast.error("Failed to create ZIP download");
    } finally {
      downloadAllRef.current = false;
      setDownloadAllLoading(false);
    }
  }, [results]);

  if (isLoading) {
    return (
      <Card className="overflow-hidden border-slate-200 shadow-md dark:border-slate-700">
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
        "animate-scale-in overflow-hidden border-0 bg-white shadow-xl dark:bg-slate-900 dark:shadow-slate-950",
        "opacity-0 [animation-fill-mode:forwards]"
      )}
      style={{ animationDelay: "100ms" }}
    >
      <CardHeader className="border-b border-slate-100 bg-gradient-to-r from-slate-100 to-white px-5 py-4 dark:border-slate-800 dark:from-slate-800 dark:to-slate-800 sm:px-6 sm:py-5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-emerald-100 shadow-inner dark:bg-emerald-900">
              <CheckCircle2 className="h-5 w-5 text-emerald-600 dark:text-emerald-400" strokeWidth={2.5} />
            </div>
            <div>
              <h2 className="font-semibold text-slate-800 dark:text-slate-100">
                Your processed images
              </h2>
              <p className="text-sm text-slate-500 dark:text-slate-400">
                {results.length} image{results.length > 1 ? "s" : ""} ready
              </p>
            </div>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={onClear}
            className="rounded-lg text-slate-500 hover:bg-slate-200 hover:text-slate-800 dark:text-slate-400 dark:hover:bg-slate-700 dark:hover:text-slate-200"
          >
            <X className="mr-1.5 h-4 w-4" />
            Clear
          </Button>
        </div>
      </CardHeader>

      <CardContent className="space-y-0 p-0">
        <div className="px-4 pt-4 sm:px-6 sm:pt-5">
          <BeforeAfterSlider
            beforeSrc={item.originalUrl}
            afterSrc={item.processedUrl}
            beforeLabel="Original"
            afterLabel="Processed"
          />
        </div>

        {results.length > 1 && (
          <div className="border-t border-slate-100 px-4 pb-4 pt-5 dark:border-slate-800 sm:px-6">
            <p className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-400 dark:text-slate-500">
              Choose image
            </p>
            <div className="relative -mx-1 flex items-center">
              <button
                type="button"
                onClick={() => setSelected((s) => (s > 0 ? s - 1 : results.length - 1))}
                className="absolute -left-1 z-10 flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-white shadow-lg ring-1 ring-slate-200 transition hover:bg-slate-50 dark:bg-slate-800 dark:ring-slate-600 dark:hover:bg-slate-700"
                aria-label="Previous"
              >
                <ChevronLeft className="h-5 w-5 text-slate-600 dark:text-slate-300" />
              </button>
              <div
                ref={scrollRef}
                className="flex flex-1 gap-3 overflow-x-auto px-11 py-2 scrollbar-thin"
              >
                {results.map((r, i) => (
                  <button
                    key={i}
                    data-index={i}
                    onClick={() => setSelected(i)}
                    className={cn(
                      "group relative shrink-0 overflow-hidden rounded-xl transition-all duration-200",
                      "h-16 w-24 sm:h-20 sm:w-28",
                      selected === i
                        ? "ring-2 ring-emerald-500 shadow-lg border-2 border-emerald-500"
                        : "opacity-70 hover:opacity-100 hover:ring-1 hover:ring-slate-300 dark:hover:ring-slate-600 border-2 border-transparent"
                    )}
                  >
                    <img
                      src={r.processedUrl}
                      alt={r.originalFilename}
                      className="h-full w-full object-cover"
                      loading="lazy"
                      draggable={false}
                    />
                    <div
                      className={cn(
                        "absolute bottom-0 left-0 right-0 bg-slate-900 px-2 py-1 text-center text-xs font-medium text-white transition",
                        selected === i ? "opacity-100" : "opacity-0 group-hover:opacity-100"
                      )}
                    >
                      {i + 1}
                    </div>
                  </button>
                ))}
              </div>
              <button
                type="button"
                onClick={() => setSelected((s) => (s < results.length - 1 ? s + 1 : 0))}
                className="absolute -right-1 z-10 flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-white shadow-lg ring-1 ring-slate-200 transition hover:bg-slate-50 dark:bg-slate-800 dark:ring-slate-600 dark:hover:bg-slate-700"
                aria-label="Next"
              >
                <ChevronRight className="h-5 w-5 text-slate-600 dark:text-slate-300" />
              </button>
            </div>
          </div>
        )}

        <div className="border-t border-slate-100 bg-slate-100 px-4 py-5 dark:border-slate-800 dark:bg-slate-800 sm:px-6 sm:py-6">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between sm:gap-6">
            <div className="min-w-0">
              <p className="truncate font-medium text-slate-800 dark:text-slate-100" title={item.originalFilename}>
                {truncateFilename(item.originalFilename, 36)}
              </p>
              <p className="mt-0.5 text-sm text-slate-500 dark:text-slate-400">
                {results.length > 1 ? `Image ${selected + 1} of ${results.length}` : "Ready to save"}
              </p>
            </div>
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:gap-3">
              <Button
                asChild
                size="lg"
                className="w-full rounded-xl bg-emerald-600 px-6 font-semibold shadow-lg transition hover:bg-emerald-700 dark:bg-emerald-600 dark:hover:bg-emerald-500 sm:w-auto"
              >
                <a
                  href={item.processedUrl}
                  download={item.processedFilename}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Download className="mr-2 h-5 w-5" />
                  Download
                </a>
              </Button>
              {results.length > 1 && (
                <Button
                  variant="outline"
                  size="lg"
                  onClick={handleDownloadAll}
                  disabled={downloadAllLoading}
                  className="w-full rounded-xl border-slate-300 font-medium transition hover:border-emerald-400 hover:bg-emerald-50 hover:text-emerald-700 dark:border-slate-600 dark:hover:border-emerald-600 dark:hover:bg-emerald-900 dark:hover:text-emerald-300 sm:w-auto"
                >
                  {downloadAllLoading ? (
                    <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                  ) : (
                    <DownloadCloud className="mr-2 h-5 w-5" />
                  )}
                  {downloadAllLoading ? "Creating ZIP…" : "Download all (ZIP)"}
                </Button>
              )}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
