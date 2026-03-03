"use client";

import { Sparkles, Trash2, AlertCircle, Loader2, Settings2 } from "lucide-react";
import { DropZone } from "./components/DropZone";
import { SelectedFilesCard } from "./components/SelectedFilesCard";
import { ResultGallery } from "./components/ResultGallery";
import { ProcessingProgress } from "./components/ProcessingProgress";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Sidebar } from "@/components/layout/Sidebar";
import { useProcessImages } from "@/hooks/useProcessImages";
import { cn } from "@/lib/utils";
import { useState } from "react";

export default function Home() {
  const {
    files,
    results,
    failed,
    isProcessing,
    progress,
    onFilesSelected,
    processImages,
    clearAll,
  } = useProcessImages();

  const [mobileOptionsOpen, setMobileOptionsOpen] = useState(false);
  const [outputFormat, setOutputFormat] = useState("png");
  const [background, setBackground] = useState("white");
  const [processingMode, setProcessingMode] = useState("standard");
  const showResultsSkeleton = isProcessing && results.length === 0 && progress.total <= 3;

  return (
    <div className="min-h-[calc(100vh-4rem)] bg-gradient-to-b from-slate-50/80 via-white to-slate-100/60 dark:from-slate-950/50 dark:via-slate-900/30 dark:to-slate-950/80">
      <div className="mx-auto max-w-6xl px-4 py-6 sm:px-6 sm:py-8 lg:px-8">
        {/* Mobile options toggle */}
        <div className="mb-4 flex items-center justify-between lg:hidden">
          <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-100">
            Image Editor
          </h2>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setMobileOptionsOpen(!mobileOptionsOpen)}
          >
            <Settings2 className="mr-1.5 h-4 w-4" />
            Options
          </Button>
        </div>
        {mobileOptionsOpen && (
          <div className="mb-6 rounded-xl border border-slate-200 bg-white p-4 dark:border-slate-700 dark:bg-slate-900/50 lg:hidden">
            <Sidebar />
          </div>
        )}

        {/* Editor section */}
        <section className="mb-12">
          <div className="mb-6">
            <h1 className="text-xl font-bold text-slate-800 dark:text-slate-100 sm:text-2xl">
              Background Removal
            </h1>
            <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
              Upload car images for AI-powered background removal. Single or batch.
            </p>
          </div>

          <DropZone
            onFilesSelected={onFilesSelected}
            selectedFiles={files}
            disabled={isProcessing}
          />

          {files.length > 0 && (
            <div className="mt-6">
              <SelectedFilesCard
                files={files}
                onClear={clearAll}
                disabled={isProcessing}
                outputFormat={outputFormat}
                onOutputFormatChange={setOutputFormat}
                background={background}
                onBackgroundChange={setBackground}
                processingMode={processingMode}
                onProcessingModeChange={setProcessingMode}
              />
            </div>
          )}

          {(files.length > 0 || results.length > 0) && (
            <div
              className={cn(
                "mt-6 flex flex-wrap gap-3 animate-fade-in",
                "opacity-0 [animation-fill-mode:forwards]"
              )}
            >
              <Button
                onClick={() => processImages()}
                disabled={isProcessing}
                size="lg"
                className="min-w-[160px] shadow-lg shadow-emerald-600/20"
              >
                {isProcessing ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Processing…
                  </>
                ) : (
                  <>
                    <Sparkles className="mr-2 h-4 w-4" />
                    Process Images
                  </>
                )}
              </Button>
              <Button
                variant="outline"
                size="lg"
                onClick={clearAll}
                disabled={isProcessing}
              >
                <Trash2 className="mr-2 h-4 w-4" />
                Clear
              </Button>
            </div>
          )}

          {isProcessing &&
            (progress.total > 0 ? (
              <div className="mt-6">
                <ProcessingProgress
                  completed={progress.completed}
                  total={progress.total}
                />
              </div>
            ) : (
              <Card
                className={cn(
                  "mt-6 animate-scale-in border-emerald-200/60 bg-gradient-to-r from-emerald-50/80 to-teal-50/50 dark:border-emerald-900/40 dark:from-emerald-950/30 dark:to-teal-950/20",
                  "opacity-0 [animation-fill-mode:forwards]"
                )}
              >
                <CardContent className="flex items-center gap-4 p-5 sm:p-6">
                  <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-emerald-100 dark:bg-emerald-900/50">
                    <Loader2 className="h-5 w-5 animate-spin text-emerald-600 dark:text-emerald-400" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="font-semibold text-slate-700 dark:text-slate-200">
                      Processing images…
                    </p>
                    <p className="text-sm text-slate-500 dark:text-slate-400">
                      AI is removing backgrounds. This may take 10–15 seconds per image.
                    </p>
                  </div>
                </CardContent>
              </Card>
            ))}

          {showResultsSkeleton && (
            <div className="mt-6 sm:mt-8">
              <ResultGallery results={[]} onClear={clearAll} isLoading />
            </div>
          )}

          {failed.length > 0 && (
            <Card
              className={cn(
                "mt-6 animate-scale-in border-amber-200/80 bg-amber-50/80 dark:border-amber-900/50 dark:bg-amber-950/30",
                "opacity-0 [animation-fill-mode:forwards]"
              )}
              style={{ animationDelay: "50ms" }}
            >
              <CardContent className="p-4 sm:p-5">
                <h3 className="flex items-center gap-2 text-sm font-semibold text-amber-800 dark:text-amber-200">
                  <AlertCircle className="h-4 w-4 shrink-0" />
                  Failed ({failed.length})
                </h3>
                <ul className="mt-2 space-y-1 text-sm text-amber-700 dark:text-amber-300">
                  {failed.map((f, i) => (
                    <li key={i}>
                      {f.filename}: {f.error}
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          )}

          {results.length > 0 && !showResultsSkeleton && (
            <div className="mt-6 sm:mt-8">
              <ResultGallery results={results} onClear={clearAll} />
            </div>
          )}
        </section>

        {/* Quick links */}
        <section className="py-12">
          <h2 className="mb-4 text-lg font-semibold text-slate-800 dark:text-slate-100">
            Learn more
          </h2>
          <div className="flex flex-wrap gap-3">
            <a
              href="/features"
              className="rounded-lg border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-700 transition hover:border-emerald-300 hover:bg-emerald-50 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300 dark:hover:border-emerald-700 dark:hover:bg-emerald-950/30"
            >
              Features
            </a>
            <a
              href="/how-it-works"
              className="rounded-lg border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-700 transition hover:border-emerald-300 hover:bg-emerald-50 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300 dark:hover:border-emerald-700 dark:hover:bg-emerald-950/30"
            >
              How it works
            </a>
            <a
              href="/api-docs"
              className="rounded-lg border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-700 transition hover:border-emerald-300 hover:bg-emerald-50 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300 dark:hover:border-emerald-700 dark:hover:bg-emerald-950/30"
            >
              API Docs
            </a>
            <a
              href="/documentation"
              className="rounded-lg border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-700 transition hover:border-emerald-300 hover:bg-emerald-50 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300 dark:hover:border-emerald-700 dark:hover:bg-emerald-950/30"
            >
              Documentation
            </a>
            <a
              href="/contact"
              className="rounded-lg border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-700 transition hover:border-emerald-300 hover:bg-emerald-50 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300 dark:hover:border-emerald-700 dark:hover:bg-emerald-950/30"
            >
              Contact
            </a>
          </div>
        </section>
      </div>
    </div>
  );
}
