"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Mail, MessageSquare, Send } from "lucide-react";
import Link from "next/link";

export default function ContactPage() {
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setSubmitted(true);
  };

  return (
    <div className="mx-auto max-w-2xl px-4 py-12 sm:px-6 lg:px-8">
      <div className="mb-12 text-center">
        <h1 className="text-3xl font-bold tracking-tight text-slate-800 dark:text-slate-100 sm:text-4xl">
          Contact
        </h1>
        <p className="mt-4 text-lg text-slate-600 dark:text-slate-400">
          Get in touch. We’re here to help.
        </p>
      </div>

      <div className="grid gap-8 sm:grid-cols-2">
        <Card className="border-slate-200/80 dark:border-slate-700/80">
          <CardHeader>
            <div className="flex items-center gap-2">
              <Mail className="h-5 w-5 text-emerald-600" />
              <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-100">
                Email
              </h2>
            </div>
          </CardHeader>
          <CardContent>
            <a
              href="mailto:support@example.com"
              className="text-emerald-600 hover:underline dark:text-emerald-400"
            >
              support@example.com
            </a>
          </CardContent>
        </Card>

        <Card className="border-slate-200/80 dark:border-slate-700/80">
          <CardHeader>
            <div className="flex items-center gap-2">
              <MessageSquare className="h-5 w-5 text-emerald-600" />
              <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-100">
                Documentation
              </h2>
            </div>
          </CardHeader>
          <CardContent>
            <Link
              href="/api-docs"
              className="text-emerald-600 hover:underline dark:text-emerald-400"
            >
              API Docs
            </Link>
            {" · "}
            <Link
              href="/documentation"
              className="text-emerald-600 hover:underline dark:text-emerald-400"
            >
              Full docs
            </Link>
          </CardContent>
        </Card>
      </div>

      <Card className="mt-8 border-slate-200/80 dark:border-slate-700/80">
        <CardHeader>
          <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-100">
            Send a message
          </h2>
        </CardHeader>
        <CardContent>
          {submitted ? (
            <p className="rounded-lg bg-emerald-50 p-4 text-sm text-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-400">
              Thanks! We’ll get back to you soon.
            </p>
          ) : (
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="space-y-2">
                <label htmlFor="contact-name" className="text-sm font-medium text-slate-700 dark:text-slate-300">
                  Name
                </label>
                <Input
                  id="contact-name"
                  type="text"
                  placeholder="Your name"
                />
              </div>
              <div className="space-y-2">
                <label htmlFor="contact-email" className="text-sm font-medium text-slate-700 dark:text-slate-300">
                  Email
                </label>
                <Input
                  id="contact-email"
                  type="email"
                  placeholder="you@example.com"
                />
              </div>
              <div className="space-y-2">
                <label htmlFor="contact-message" className="text-sm font-medium text-slate-700 dark:text-slate-300">
                  Message
                </label>
                <Textarea
                  id="contact-message"
                  rows={4}
                  placeholder="How can we help?"
                />
              </div>
              <Button type="submit">
                <Send className="mr-2 h-4 w-4" />
                Send
              </Button>
            </form>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
