"use client";

import {
  ImageIcon,
  Trash2,
  Layers,
  Palette,
  Settings2,
  FileImage,
} from "lucide-react";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { cn } from "@/lib/utils";

function formatFileSize(bytes: number) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

type Props = {
  files: File[];
  onClear: () => void;
  disabled?: boolean;
  outputFormat: string;
  onOutputFormatChange: (v: string) => void;
  background: string;
  onBackgroundChange: (v: string) => void;
  processingMode: string;
  onProcessingModeChange: (v: string) => void;
};

export function SelectedFilesCard({
  files,
  onClear,
  disabled,
  outputFormat,
  onOutputFormatChange,
  background,
  onBackgroundChange,
  processingMode,
  onProcessingModeChange,
}: Props) {
  if (files.length === 0) return null;

  return (
    <Card
      className={cn(
        "animate-scale-in overflow-hidden border-slate-200/80 shadow-md dark:border-slate-700/80",
        "opacity-0 [animation-fill-mode:forwards]"
      )}
      style={{ animationDelay: "50ms" }}
    >
      <CardHeader className="flex flex-row items-center justify-between gap-4 border-b border-slate-100 pb-4 dark:border-slate-800">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-100 dark:bg-emerald-900/50">
            <ImageIcon className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
          </div>
          <div>
            <h3 className="font-semibold text-slate-800 dark:text-slate-100">
              {files.length} file{files.length > 1 ? "s" : ""} selected
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              {formatFileSize(files.reduce((a, f) => a + f.size, 0))} total
            </p>
          </div>
        </div>

        <AlertDialog>
          <AlertDialogTrigger asChild>
            <Button
              variant="outline"
              size="sm"
              disabled={disabled}
              className="text-red-600 hover:bg-red-50 hover:text-red-700 dark:text-red-400 dark:hover:bg-red-950/50"
            >
              <Trash2 className="mr-2 h-4 w-4" />
              Clear all
            </Button>
          </AlertDialogTrigger>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Clear all files?</AlertDialogTitle>
              <AlertDialogDescription>
                This will remove all {files.length} selected file{files.length > 1 ? "s" : ""} from
                the queue. You can upload new images afterward.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>Cancel</AlertDialogCancel>
              <AlertDialogAction onClick={onClear}>Clear all</AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </CardHeader>

      <CardContent className="space-y-6 p-5">
        {/* Processing options - Shadcn dropdowns */}
        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <Settings2 className="h-4 w-4 text-emerald-600" />
            <h4 className="text-sm font-semibold text-slate-800 dark:text-slate-100">
              Processing Options
            </h4>
          </div>

          <div className="grid gap-4 sm:grid-cols-3">
            <div>
              <label className="mb-1.5 flex items-center gap-2 text-xs font-medium text-slate-600 dark:text-slate-400">
                <Layers className="h-3.5 w-3.5" />
                Output Format
              </label>
              <Select value={outputFormat} onValueChange={onOutputFormatChange}>
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="png">PNG (transparency)</SelectItem>
                  <SelectItem value="jpeg">JPEG (white background)</SelectItem>
                  <SelectItem value="webp">WebP (smaller size)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <label className="mb-1.5 flex items-center gap-2 text-xs font-medium text-slate-600 dark:text-slate-400">
                <Palette className="h-3.5 w-3.5" />
                Background
              </label>
              <Select value={background} onValueChange={onBackgroundChange}>
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="white">White (studio)</SelectItem>
                  <SelectItem value="transparent">Transparent</SelectItem>
                  <SelectItem value="light-gray">Light gray</SelectItem>
                  <SelectItem value="dark-gray">Dark gray</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <label className="mb-1.5 flex items-center gap-2 text-xs font-medium text-slate-600 dark:text-slate-400">
                <Settings2 className="h-3.5 w-3.5" />
                Processing Mode
              </label>
              <Select value={processingMode} onValueChange={onProcessingModeChange}>
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="standard">Standard</SelectItem>
                  <SelectItem value="fast">Fast</SelectItem>
                  <SelectItem value="high-quality">High quality</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>

        {/* File list */}
        <div>
          <h4 className="mb-3 text-xs font-medium text-slate-500 dark:text-slate-400">
            Selected files
          </h4>
          <ul className="max-h-40 space-y-2 overflow-y-auto rounded-lg border border-slate-200 bg-slate-50/50 p-3 dark:border-slate-700 dark:bg-slate-800/30">
            {files.map((f, i) => (
              <li
                key={i}
                className="flex items-center gap-3 rounded-lg px-3 py-2 text-sm text-slate-700 dark:text-slate-300"
              >
                <FileImage className="h-4 w-4 shrink-0 text-slate-400" />
                <span className="min-w-0 flex-1 truncate" title={f.name}>
                  {f.name}
                </span>
                <span className="shrink-0 text-xs text-slate-500 dark:text-slate-400">
                  {formatFileSize(f.size)}
                </span>
              </li>
            ))}
          </ul>
        </div>
      </CardContent>
    </Card>
  );
}
