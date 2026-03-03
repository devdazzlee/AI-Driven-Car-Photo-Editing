import { Card, CardContent, CardHeader } from "@/components/ui/card";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { FileCode, ArrowRight } from "lucide-react";

const endpoints = [
  {
    method: "POST",
    path: "/api/process",
    description: "Upload one or more images for background removal. Returns result inline for 1–3 images, or job_id for batch (4+).",
  },
  {
    method: "GET",
    path: "/api/status/{job_id}",
    description: "Poll batch job progress. Returns total, completed, failed, results, and status.",
  },
  {
    method: "GET",
    path: "/api/download/{job_id}/{filename}",
    description: "Download a processed image by job_id and filename (e.g. image_processed.png).",
  },
];

export default function ApiDocsPage() {
  return (
    <div className="mx-auto max-w-4xl px-4 py-12 sm:px-6 lg:px-8">
      <div className="mb-12">
        <h1 className="text-3xl font-bold tracking-tight text-slate-800 dark:text-slate-100 sm:text-4xl">
          API Documentation
        </h1>
        <p className="mt-4 text-lg text-slate-600 dark:text-slate-400">
          REST API for integrating Car Image AI into your workflow.
        </p>
      </div>

      <div className="space-y-8">
        <Card className="border-slate-200/80 dark:border-slate-700/80">
          <CardHeader>
            <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-100">
              Base URL
            </h2>
          </CardHeader>
          <CardContent>
            <code className="block rounded-lg bg-slate-100 px-4 py-3 font-mono text-sm dark:bg-slate-800">
              https://your-domain.com/api
            </code>
            <p className="mt-2 text-sm text-slate-500 dark:text-slate-400">
              Replace with your deployment URL (e.g. https://your-domain.com/api)
            </p>
          </CardContent>
        </Card>

        <div>
          <h2 className="mb-4 text-xl font-semibold text-slate-800 dark:text-slate-100">
            Endpoints
          </h2>
          <div className="space-y-6">
            {endpoints.map((ep) => (
              <Card key={ep.path} className="border-slate-200/80 dark:border-slate-700/80">
                <CardContent className="pt-6">
                  <div className="flex flex-wrap items-center gap-2">
                    <span
                      className={`rounded px-2 py-0.5 text-xs font-bold ${
                        ep.method === "POST"
                          ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/50 dark:text-emerald-400"
                          : "bg-blue-100 text-blue-700 dark:bg-blue-900/50 dark:text-blue-400"
                      }`}
                    >
                      {ep.method}
                    </span>
                    <code className="font-mono text-sm">{ep.path}</code>
                  </div>
                  <p className="mt-3 text-sm text-slate-600 dark:text-slate-400">
                    {ep.description}
                  </p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>

        <Card className="border-slate-200/80 dark:border-slate-700/80">
          <CardHeader>
            <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-100">
              Example: Upload & Process
            </h2>
          </CardHeader>
          <CardContent>
            <pre className="overflow-x-auto rounded-lg bg-slate-900 p-4 text-sm text-slate-100">
{`curl -X POST /api/process \\
  -F "files=@car1.jpg" \\
  -F "files=@car2.jpg"`}
            </pre>
          </CardContent>
        </Card>
      </div>

      <div className="mt-12 flex gap-4">
        <Button asChild>
          <Link href="/">
            Open Editor
            <ArrowRight className="ml-2 h-4 w-4" />
          </Link>
        </Button>
        <Button variant="outline" asChild>
          <Link href="/documentation">Full Documentation</Link>
        </Button>
      </div>
    </div>
  );
}
