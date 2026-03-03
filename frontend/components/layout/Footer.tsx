"use client";

import Link from "next/link";
import { Github, FileText, Mail, ExternalLink } from "lucide-react";
import { AIIcon } from "@/components/ui/AIIcon";

const footerLinks = {
  Product: [
    { label: "Editor", href: "/" },
    { label: "Features", href: "/features" },
    { label: "API Docs", href: "/api-docs" },
  ],
  Resources: [
    { label: "How it works", href: "/how-it-works" },
    { label: "RMBG Model", href: "https://huggingface.co/briaai/RMBG-1.4", external: true },
    { label: "Hugging Face", href: "https://huggingface.co", external: true },
  ],
  Support: [
    { label: "Contact", href: "/contact" },
    { label: "Documentation", href: "/documentation" },
  ],
};

export function Footer() {
  return (
    <footer className="mt-auto border-t border-slate-200 bg-slate-50 dark:border-slate-800 dark:bg-slate-900/50">
      <div className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8">
        <div className="grid gap-8 sm:grid-cols-2 lg:grid-cols-5">
          <div className="lg:col-span-2">
            <Link href="/" className="flex items-center gap-2">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-violet-500 to-indigo-600 text-white">
                <AIIcon className="text-white" size={20} />
              </div>
              <span className="text-lg font-bold text-slate-800 dark:text-slate-100">
                Car Image AI
              </span>
            </Link>
            <p className="mt-3 max-w-xs text-sm text-slate-600 dark:text-slate-400">
              AI-powered car photo editing. Remove backgrounds, standardize lighting, and
              process images in batch with one click.
            </p>
          </div>

          <div>
            <h4 className="text-sm font-semibold text-slate-800 dark:text-slate-100">
              Product
            </h4>
            <ul className="mt-4 space-y-3">
              {footerLinks.Product.map((link) => (
                <li key={link.href}>
                  <Link
                    href={link.href}
                    className="flex items-center gap-1.5 text-sm text-slate-600 transition-colors hover:text-emerald-600 dark:text-slate-400 dark:hover:text-emerald-400"
                  >
                    {link.label}
                    <ExternalLink className="h-3 w-3" />
                  </Link>
                </li>
              ))}
            </ul>
          </div>
          <div>
            <h4 className="text-sm font-semibold text-slate-800 dark:text-slate-100">
              Resources
            </h4>
            <ul className="mt-4 space-y-3">
              {footerLinks.Resources.map((link) => (
                <li key={link.href}>
                  <Link
                    href={link.href}
                    target={link.external ? "_blank" : undefined}
                    rel={link.external ? "noopener noreferrer" : undefined}
                    className="flex items-center gap-1.5 text-sm text-slate-600 transition-colors hover:text-emerald-600 dark:text-slate-400 dark:hover:text-emerald-400"
                  >
                    {link.label}
                    <ExternalLink className="h-3 w-3" />
                  </Link>
                </li>
              ))}
            </ul>
          </div>
          <div>
            <h4 className="text-sm font-semibold text-slate-800 dark:text-slate-100">
              Support
            </h4>
            <ul className="mt-4 space-y-3">
              {footerLinks.Support.map((link) => (
                <li key={link.href}>
                  <Link
                    href={link.href}
                    className="flex items-center gap-1.5 text-sm text-slate-600 transition-colors hover:text-emerald-600 dark:text-slate-400 dark:hover:text-emerald-400"
                  >
                    {link.label}
                    <ExternalLink className="h-3 w-3" />
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        </div>

        <div className="mt-12 flex flex-col items-center justify-between gap-4 border-t border-slate-200 pt-8 dark:border-slate-700 sm:flex-row">
          <p className="text-sm text-slate-500 dark:text-slate-400">
            © {new Date().getFullYear()} Car Image AI. Built for dealerships.
          </p>
          <div className="flex items-center gap-4">
            <Link
              href="#"
              className="text-slate-400 transition-colors hover:text-slate-600 dark:hover:text-slate-300"
              aria-label="GitHub"
            >
              <Github className="h-5 w-5" />
            </Link>
            <Link
              href="/api-docs"
              className="text-slate-400 transition-colors hover:text-slate-600 dark:hover:text-slate-300"
              aria-label="Documentation"
            >
              <FileText className="h-5 w-5" />
            </Link>
            <Link
              href="/contact"
              className="text-slate-400 transition-colors hover:text-slate-600 dark:hover:text-slate-300"
              aria-label="Contact"
            >
              <Mail className="h-5 w-5" />
            </Link>
          </div>
        </div>
      </div>
    </footer>
  );
}
