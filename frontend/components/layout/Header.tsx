"use client";

import Link from "next/link";
import {
  ImageIcon,
  HelpCircle,
  FileText,
  Zap,
  Menu,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { AIIcon } from "@/components/ui/AIIcon";
import { useState } from "react";

const navItems = [
  { href: "/", label: "Editor", icon: ImageIcon },
  { href: "/features", label: "Features", icon: Zap },
  { href: "/how-it-works", label: "How it works", icon: HelpCircle },
  { href: "/api-docs", label: "API Docs", icon: FileText },
];

export function Header() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <header className="sticky top-0 z-50 w-full border-b border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-slate-950">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between gap-4 px-4 sm:px-6 lg:px-8">
        <Link
          href="/"
          className="flex shrink-0 items-center gap-2.5 transition-opacity hover:opacity-90"
        >
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-violet-500 to-indigo-600 text-white shadow-md">
            <AIIcon className="text-white" size={20} />
          </div>
          <div className="hidden sm:block">
            <span className="text-base font-bold text-slate-800 dark:text-slate-100">
              Car Image AI
            </span>
            <span className="ml-1.5 text-xs text-slate-500 dark:text-slate-400">
              Pro
            </span>
          </div>
        </Link>

        <nav className="hidden items-center gap-1 md:flex">
          {navItems.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className="flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium text-slate-600 transition-colors hover:bg-slate-100 hover:text-slate-900 dark:text-slate-400 dark:hover:bg-slate-800 dark:hover:text-slate-100"
            >
              <item.icon className="h-4 w-4" />
              {item.label}
            </Link>
          ))}
        </nav>

        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" asChild className="hidden sm:inline-flex">
            <Link href="/api-docs">
              <FileText className="mr-1.5 h-4 w-4" />
              Docs
            </Link>
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="md:hidden"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            aria-label="Toggle menu"
          >
            {mobileMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </Button>
        </div>
      </div>

      {mobileMenuOpen && (
        <div className="border-t border-slate-200 dark:border-slate-800 md:hidden">
          <nav className="flex flex-col gap-0.5 p-3">
            {navItems.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setMobileMenuOpen(false)}
                className="flex items-center gap-2 rounded-lg px-3 py-2.5 text-sm font-medium text-slate-600 transition-colors hover:bg-slate-100 dark:text-slate-400 dark:hover:bg-slate-800"
              >
                <item.icon className="h-4 w-4" />
                {item.label}
              </Link>
            ))}
          </nav>
        </div>
      )}
    </header>
  );
}
