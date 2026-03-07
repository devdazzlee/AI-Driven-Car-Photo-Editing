import { Card, CardContent } from "@/components/ui/card";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Upload, Sparkles, Eye, Download } from "lucide-react";

const steps = [
  { step: 1, icon: Upload, title: "Upload", description: "Drag & drop your raw car photos or click to browse. Supports JPEG, PNG, WebP, NEF. Single or batch (up to 50 images)." },
  { step: 2, icon: Sparkles, title: "Process", description: "Click Process Images. By default, floor, walls and corner are kept—same color, natural look. No deletion of floors or walls. Optional: enhance car & lighting, or remove background for studio style." },
  { step: 3, icon: Eye, title: "Preview", description: "Use the before/after slider to compare original and processed images. Verify quality before download." },
  { step: 4, icon: Download, title: "Download", description: "Download processed images. Floor and walls intact with corner visible—ready for your website, listings, or marketing." },
];

export default function HowItWorksPage() {
  return (
    <div className="mx-auto max-w-4xl px-4 py-12 sm:px-6 lg:px-8">
      <div className="mb-12 text-center">
        <h1 className="text-3xl font-bold tracking-tight text-slate-800 dark:text-slate-100 sm:text-4xl">
          How it works
        </h1>
        <p className="mt-4 text-lg text-slate-600 dark:text-slate-400">
          Four simple steps from upload to download.
        </p>
      </div>

      <div className="space-y-8">
        {steps.map((s) => (
          <Card key={s.step} className="border-slate-200/80 dark:border-slate-700/80">
            <CardContent className="flex flex-col gap-4 p-6 sm:flex-row sm:items-start sm:gap-6">
              <div className="flex shrink-0 items-center gap-4">
                <span className="flex h-12 w-12 items-center justify-center rounded-full bg-emerald-100 text-lg font-bold text-emerald-600 dark:bg-emerald-900/50 dark:text-emerald-400">
                  {s.step}
                </span>
                <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-slate-100 dark:bg-slate-800">
                  <s.icon className="h-6 w-6 text-slate-600 dark:text-slate-400" />
                </div>
              </div>
              <div>
                <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-100">
                  {s.title}
                </h2>
                <p className="mt-1 text-slate-600 dark:text-slate-400">{s.description}</p>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="mt-16 flex justify-center">
        <Button size="lg" asChild>
          <Link href="/">Get started</Link>
        </Button>
      </div>
    </div>
  );
}
