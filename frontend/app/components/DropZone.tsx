"use client";

import { useCallback, useRef, useState } from "react";
import { Upload, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

type Props = {
  onFilesSelected: (files: File[]) => void;
  selectedFiles: File[];
  disabled?: boolean;
};

const ALLOWED = ["image/jpeg", "image/png", "image/webp"];
const MAX_SIZE_MB = 20;

export function DropZone({
  onFilesSelected,
  selectedFiles,
  disabled,
}: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const filterFiles = useCallback((list: FileList | null): File[] => {
    if (!list) return [];
    return Array.from(list).filter((f) => {
      if (!ALLOWED.includes(f.type)) return false;
      if (f.size > MAX_SIZE_MB * 1024 * 1024) return false;
      return true;
    });
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setIsDragging(false);
      if (disabled) return;
      const files = filterFiles(e.dataTransfer.files);
      if (files.length) onFilesSelected(files);
    },
    [disabled, filterFiles, onFilesSelected]
  );

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => setIsDragging(false), []);

  const handleClick = useCallback(() => {
    if (disabled) return;
    inputRef.current?.click();
  }, [disabled]);

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = filterFiles(e.target.files);
      if (files.length) onFilesSelected(files);
      e.target.value = "";
    },
    [filterFiles, onFilesSelected]
  );

  return (
    <div className="space-y-4">
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={handleClick}
        className={cn(
          "group relative cursor-pointer overflow-hidden rounded-2xl border-2 border-dashed p-8 text-center transition-all duration-300 sm:p-12",
          "animate-fade-in",
          disabled
            ? "cursor-not-allowed border-slate-200 bg-slate-100/80 dark:border-slate-700 dark:bg-slate-800/50"
            : isDragging
              ? "scale-[1.02] border-emerald-400 bg-emerald-50/80 shadow-lg shadow-emerald-500/10 dark:border-emerald-600 dark:bg-emerald-950/30"
              : "border-slate-300 bg-white hover:border-emerald-300 hover:bg-emerald-50/30 hover:shadow-md dark:border-slate-600 dark:bg-slate-900/50 dark:hover:border-emerald-700 dark:hover:bg-emerald-950/20"
        )}
      >
        {disabled && (
          <div className="absolute inset-0 z-20 flex items-center justify-center bg-white/80 backdrop-blur-sm dark:bg-slate-900/80">
            <div className="flex flex-col items-center gap-2">
              <Loader2 className="h-8 w-8 animate-spin text-emerald-600" />
              <span className="text-sm font-medium text-slate-600 dark:text-slate-400">
                Processing…
              </span>
            </div>
          </div>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp"
          multiple
          className="hidden"
          onChange={handleChange}
          disabled={disabled}
        />
        <div className="flex flex-col items-center gap-4 sm:flex-row sm:justify-center sm:gap-6">
          <div
            className={cn(
              "flex h-16 w-16 shrink-0 items-center justify-center rounded-2xl transition-all duration-300 sm:h-20 sm:w-20",
              isDragging && !disabled
                ? "bg-emerald-500 text-white"
                : "bg-slate-100 text-slate-500 group-hover:bg-emerald-100 group-hover:text-emerald-600 dark:bg-slate-800 dark:group-hover:bg-emerald-900/50"
            )}
          >
            <Upload className="h-8 w-8 sm:h-10 sm:w-10" strokeWidth={2} />
          </div>
          <div className="space-y-1">
            <p className="text-base font-semibold text-slate-700 dark:text-slate-200 sm:text-lg">
              Drag & drop car images here
            </p>
          <p className="text-sm text-slate-500 dark:text-slate-400">
            or click to browse • JPEG, PNG, WebP (max {MAX_SIZE_MB}MB each) • Single or batch
          </p>
          </div>
        </div>
      </div>

    </div>
  );
}
