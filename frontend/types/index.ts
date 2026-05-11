export type ProcessedItem = {
  originalFilename: string;
  processedFilename: string;
  originalUrl: string;
  processedUrl: string;
  // After a refine, holds the URL of the image as it was BEFORE this refine.
  // Used by the before/after slider so the user sees "what the refine changed",
  // not "what the whole pipeline changed since the raw upload".
  // Undefined on the initial process result — the slider falls back to originalUrl.
  previousProcessedUrl?: string;
  previousProcessedFilename?: string;
  success: boolean;
  jobId: string;
};
