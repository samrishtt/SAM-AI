import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SAM-AI — Sovereign AI Research Platform",
  description: "Open-weight artificial intelligence research on verifiable reasoning, code intelligence, and post-training dynamics by Parallax Intelligence Lab.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-[#FAF8F5] text-[#28231F] antialiased selection:bg-[#FBEBDD] selection:text-[#B95F34]">
        {children}
      </body>
    </html>
  );
}
