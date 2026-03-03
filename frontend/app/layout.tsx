import type { Metadata } from "next";
import { Plus_Jakarta_Sans } from "next/font/google";
import { Toaster } from "sonner";
import "./globals.css";
import { Header } from "@/components/layout/Header";
import { Sidebar } from "@/components/layout/Sidebar";
import { Footer } from "@/components/layout/Footer";

const plusJakarta = Plus_Jakarta_Sans({
  variable: "--font-plus-jakarta",
  subsets: ["latin"],
  display: "swap",
  weight: ["400", "500", "600", "700"],
});

export const metadata: Metadata = {
  title: "Car Image AI | Background Removal",
  description: "AI-powered car photo editing. Upload images for automatic background removal. Single or batch processing.",
  keywords: ["car photos", "background removal", "AI editing", "image processing"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={plusJakarta.variable}>
      <body className="flex min-h-screen flex-col font-sans antialiased">
        <Toaster position="top-center" richColors closeButton toastOptions={{ duration: 4000 }} />
        <Header />
        <div className="flex flex-1 flex-col lg:flex-row">
          <aside className="hidden shrink-0 lg:block lg:w-72 lg:border-r lg:border-slate-200 lg:dark:border-slate-800">
            <div className="sticky top-16 h-[calc(100vh-4rem)] overflow-y-auto p-4">
              <Sidebar />
            </div>
          </aside>
          <main className="min-w-0 flex-1">
            {children}
          </main>
        </div>
        <Footer />
      </body>
    </html>
  );
}
