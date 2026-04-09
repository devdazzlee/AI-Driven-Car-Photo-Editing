"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import type { ProcessedItem } from "@/types";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type ProcessOptions = {
  outputFormat: string;
  background: string;
  processingMode?: string;
  lightingBoost?: number;
};

type UseProcessImagesReturn = {
  files: File[];
  results: ProcessedItem[];
  failed: { filename: string; error: string }[];
  isProcessing: boolean;
  progress: { completed: number; total: number; jobId: string };
  onFilesSelected: (files: File[]) => void;
  processImages: (options?: ProcessOptions) => Promise<void>;
  refineImage: (item: ProcessedItem, feedback: string, outputFormat: string) => Promise<void>;
  clearAll: () => void;
};

export function useProcessImages(): UseProcessImagesReturn {
  const [files, setFiles] = useState<File[]>([]);
  const [results, setResults] = useState<ProcessedItem[]>([]);
  const [failed, setFailed] = useState<{ filename: string; error: string }[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState({ completed: 0, total: 0, jobId: "" });
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const objectUrlRefs = useRef<string[]>([]);

  const revokeObjectUrls = useCallback(() => {
    objectUrlRefs.current.forEach((url) => URL.revokeObjectURL(url));
    objectUrlRefs.current = [];
  }, []);

  const onFilesSelected = useCallback((selected: File[]) => {
    setFiles(selected);
    setResults([]);
    setFailed([]);
    revokeObjectUrls();
  }, [revokeObjectUrls]);

  const clearAll = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setFiles([]);
    setResults([]);
    setFailed([]);
    setProgress({ completed: 0, total: 0, jobId: "" });
    revokeObjectUrls();
  }, [revokeObjectUrls]);

  useEffect(() => () => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    revokeObjectUrls();
  }, [revokeObjectUrls]);

  const processImages = useCallback(
    async (options?: ProcessOptions) => {
      if (files.length === 0) return;

      setIsProcessing(true);
      setResults([]);
      setFailed([]);
      revokeObjectUrls();

      const formData = new FormData();
      files.forEach((f) => formData.append("files", f));
      if (options?.outputFormat) formData.append("output_format", options.outputFormat);
      if (options?.background) formData.append("background", options.background);
      if (options?.processingMode) formData.append("processing_mode", options.processingMode);
      if (typeof options?.lightingBoost === "number") {
        formData.append("lighting_boost", String(options.lightingBoost));
      }

      try {
        const res = await fetch(`${API_URL}/api/process`, {
          method: "POST",
          body: formData,
        });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        const d = err.detail;
        const msg =
          typeof d === "string"
            ? d
            : Array.isArray(d)
              ? d[0]?.msg || JSON.stringify(d)
              : JSON.stringify(d || "Processing failed");
        throw new Error(msg);
      }

      const data = await res.json();

      if (data.job_id && data.total > 3) {
        setProgress({ completed: 0, total: data.total, jobId: data.job_id });

        intervalRef.current = setInterval(async () => {
          try {
            const statusRes = await fetch(`${API_URL}/api/status/${data.job_id}`);
            const status = await statusRes.json();
            setProgress((p) => ({ ...p, completed: status.completed }));

            if (status.status === "completed") {
              if (intervalRef.current) {
                clearInterval(intervalRef.current);
                intervalRef.current = null;
              }
              setIsProcessing(false);

              const items: ProcessedItem[] = (status.results || []).map(
                (r: { original_filename: string; processed_filename: string; original_preview?: string }) => {
                  let originalUrl: string;
                  if (r.original_preview) {
                    originalUrl = `${API_URL}/api/download/${data.job_id}/${r.original_preview}`;
                  } else {
                    const orig = files.find((f) => f.name === r.original_filename);
                    originalUrl = orig ? URL.createObjectURL(orig) : "";
                    if (originalUrl) objectUrlRefs.current.push(originalUrl);
                  }
                  return {
                    originalFilename: r.original_filename,
                    processedFilename: r.processed_filename,
                    originalUrl,
                    processedUrl: `${API_URL}/api/download/${data.job_id}/${r.processed_filename}`,
                    success: true,
                    jobId: data.job_id,
                  };
                }
              );
              setResults(items);
              setFailed(status.failed || []);
              setProgress({ completed: 0, total: 0, jobId: "" });
              const failCount = (status.failed || []).length;
              if (failCount > 0) {
                toast.warning(`${items.length} processed, ${failCount} failed`);
              } else {
                toast.success(`${items.length} image${items.length > 1 ? "s" : ""} processed successfully`);
              }
            }
          } catch {
            // Poll error - keep trying
          }
        }, 1500);
      } else {
        const jobId = data.job_id;
        const items: ProcessedItem[] = (data.results || []).map(
          (r: { original_filename: string; processed_filename: string; original_preview?: string }) => {
            let originalUrl: string;
            if (r.original_preview) {
              originalUrl = `${API_URL}/api/download/${jobId}/${r.original_preview}`;
            } else {
              const orig = files.find((f) => f.name === r.original_filename);
              originalUrl = orig ? URL.createObjectURL(orig) : "";
              if (originalUrl) objectUrlRefs.current.push(originalUrl);
            }
            return {
              originalFilename: r.original_filename,
              processedFilename: r.processed_filename,
              originalUrl,
              processedUrl: `${API_URL}/api/download/${jobId}/${r.processed_filename}`,
              success: true,
              jobId,
            };
          }
        );
        setResults(items);
        setFailed(data.failed || []);
        setIsProcessing(false);
        const failCount = (data.failed || []).length;
        if (failCount > 0) {
          toast.warning(`${items.length} processed, ${failCount} failed`);
        } else {
          toast.success(`${items.length} image${items.length > 1 ? "s" : ""} processed successfully`);
        }
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setFailed([{ filename: "request", error: message }]);
      setIsProcessing(false);
      toast.error(message);
    }
  },
    [files, revokeObjectUrls]
  );

  const refineImage = useCallback(
    async (item: ProcessedItem, feedback: string, outputFormat: string) => {
      const fetchOpts: RequestInit = { cache: "no-store" };
      // Fetch both images as blobs (avoid stale cache; CORS must allow frontend origin on API)
      const [origBlob, procBlob] = await Promise.all([
        fetch(item.originalUrl, fetchOpts).then((r) => {
          if (!r.ok) throw new Error(`Could not load original image (${r.status})`);
          return r.blob();
        }),
        fetch(item.processedUrl, fetchOpts).then((r) => {
          if (!r.ok) throw new Error(`Could not load processed image (${r.status})`);
          return r.blob();
        }),
      ]);

      const formData = new FormData();
      formData.append("original_file", new File([origBlob], item.originalFilename));
      formData.append("processed_file", new File([procBlob], item.processedFilename));
      formData.append("feedback", feedback);
      formData.append("job_id", item.jobId);
      formData.append("output_format", outputFormat);

      const res = await fetch(`${API_URL}/api/refine`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        const d = err.detail;
        const msg =
          typeof d === "string"
            ? d
            : Array.isArray(d)
              ? d[0]?.msg || JSON.stringify(d)
              : JSON.stringify(d || "Refinement failed");
        throw new Error(msg);
      }

      const data = await res.json();
      // Cache-bust so the before/after view always loads the new PNG (same path was overwriting before)
      const refinedUrl = `${API_URL}${data.download_url}?v=${Date.now()}`;

      // Match by job + original name so re-refine after filename changes to *_refined still updates the row
      setResults((prev) =>
        prev.map((r) =>
          r.jobId === item.jobId && r.originalFilename === item.originalFilename
            ? { ...r, processedUrl: refinedUrl, processedFilename: data.refined_filename }
            : r
        )
      );
    },
    []
  );

  return {
    files,
    results,
    failed,
    isProcessing,
    progress,
    onFilesSelected,
    processImages,
    refineImage,
    clearAll,
  };
}
