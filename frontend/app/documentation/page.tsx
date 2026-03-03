import { Card, CardContent } from "@/components/ui/card";
import Link from "next/link";
import { FileText, Code, Book, ExternalLink } from "lucide-react";

const docSections = [
  {
    title: "API Reference",
    description: "REST endpoints, request/response formats, and examples.",
    href: "/api-docs",
    icon: Code,
  },
  {
    title: "How it works",
    description: "Step-by-step guide from upload to download.",
    href: "/how-it-works",
    icon: Book,
  },
  {
    title: "RMBG-1.4 Model",
    description: "Background removal model by BRIA AI on Hugging Face.",
    href: "https://huggingface.co/briaai/RMBG-1.4",
    external: true,
    icon: ExternalLink,
  },
  {
    title: "Hugging Face",
    description: "Model hosting and inference platform.",
    href: "https://huggingface.co",
    external: true,
    icon: ExternalLink,
  },
];

export default function DocumentationPage() {
  return (
    <div className="mx-auto max-w-4xl px-4 py-12 sm:px-6 lg:px-8">
      <div className="mb-12">
        <h1 className="text-3xl font-bold tracking-tight text-slate-800 dark:text-slate-100 sm:text-4xl">
          Documentation
        </h1>
        <p className="mt-4 text-lg text-slate-600 dark:text-slate-400">
          Everything you need to use Car Image AI.
        </p>
      </div>

      <div className="grid gap-6 sm:grid-cols-2">
        {docSections.map((section) => (
          <Link
            key={section.href}
            href={section.href}
            target={section.external ? "_blank" : undefined}
            rel={section.external ? "noopener noreferrer" : undefined}
            className="block"
          >
            <Card className="h-full border-slate-200/80 transition-shadow hover:shadow-md dark:border-slate-700/80">
              <CardContent className="flex items-start gap-4 p-6">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-emerald-100 dark:bg-emerald-900/50">
                  <section.icon className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
                </div>
                <div>
                  <h2 className="font-semibold text-slate-800 dark:text-slate-100">
                    {section.title}
                    {section.external && (
                      <ExternalLink className="ml-1 inline h-3.5 w-3.5" />
                    )}
                  </h2>
                  <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">
                    {section.description}
                  </p>
                </div>
              </CardContent>
            </Card>
          </Link>
        ))}
      </div>

      <Card className="mt-8 border-slate-200/80 dark:border-slate-700/80">
        <CardContent className="flex items-center justify-between gap-4 p-6">
          <div className="flex items-center gap-3">
            <FileText className="h-8 w-8 text-emerald-600" />
            <div>
              <h3 className="font-semibold text-slate-800 dark:text-slate-100">
                Quick start
              </h3>
              <p className="text-sm text-slate-600 dark:text-slate-400">
                Upload images and process in the editor
              </p>
            </div>
          </div>
          <Link
            href="/"
            className="shrink-0 rounded-lg bg-emerald-600 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-500"
          >
            Open Editor
          </Link>
        </CardContent>
      </Card>
    </div>
  );
}
