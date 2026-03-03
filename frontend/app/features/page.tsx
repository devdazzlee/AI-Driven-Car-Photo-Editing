import {
  Zap,
  Layers,
  ImageIcon,
  Clock,
  Download,
  CheckCircle2,
} from "lucide-react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import Link from "next/link";
import { Button } from "@/components/ui/button";

const features = [
  {
    icon: Layers,
    title: "Batch Processing",
    description: "Process up to 50 car images in a single batch. Upload once, process all with one click.",
  },
  {
    icon: Zap,
    title: "95% Accuracy",
    description: "RMBG-1.4 delivers studio-quality background removal specifically optimized for automotive photography.",
  },
  {
    icon: ImageIcon,
    title: "Before/After Preview",
    description: "Interactive slider to compare original and processed images side by side before download.",
  },
  {
    icon: Clock,
    title: "10–15 Sec per Image",
    description: "AI-powered processing takes seconds, not hours. Scale your workflow without scaling your team.",
  },
  {
    icon: Download,
    title: "Flexible Output",
    description: "Export as PNG with transparency or JPEG with white/studio background. Your choice.",
  },
  {
    icon: CheckCircle2,
    title: "Failed Image Handling",
    description: "Automatically flag failed images. Processing logs track every job for debugging.",
  },
];

export default function FeaturesPage() {
  return (
    <div className="mx-auto max-w-5xl px-4 py-12 sm:px-6 lg:px-8">
      <div className="mb-12 text-center">
        <h1 className="text-3xl font-bold tracking-tight text-slate-800 dark:text-slate-100 sm:text-4xl">
          Features
        </h1>
        <p className="mt-4 text-lg text-slate-600 dark:text-slate-400">
          Everything you need to automate car photo editing at scale.
        </p>
      </div>

      <div className="grid gap-8 sm:grid-cols-2 lg:grid-cols-3">
        {features.map((feature) => (
          <Card
            key={feature.title}
            className="border-slate-200/80 transition-shadow hover:shadow-md dark:border-slate-700/80"
          >
            <CardHeader>
              <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-emerald-100 dark:bg-emerald-900/50">
                <feature.icon className="h-6 w-6 text-emerald-600 dark:text-emerald-400" />
              </div>
              <h2 className="mt-4 text-lg font-semibold text-slate-800 dark:text-slate-100">
                {feature.title}
              </h2>
            </CardHeader>
            <CardContent>
              <p className="text-slate-600 dark:text-slate-400">{feature.description}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="mt-16 flex justify-center">
        <Button size="lg" asChild>
          <Link href="/">
            Open Editor
          </Link>
        </Button>
      </div>
    </div>
  );
}
