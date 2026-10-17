"use client";

import React, { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { MessageSquare, BookOpen, Activity, Github, Menu, X, ExternalLink, Shield } from "lucide-react";

export const Navbar: React.FC = () => {
  const pathname = usePathname();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navLinks = [
    { href: "/", label: "Overview", icon: Shield },
    { href: "/chat", label: "Chat & Code Demo", icon: MessageSquare },
    { href: "/research", label: "Research & Audits", icon: BookOpen },
    { href: "/status", label: "System Status", icon: Activity },
  ];

  const isActive = (href: string) => {
    if (href === "/") return pathname === "/";
    return pathname.startsWith(href);
  };

  return (
    <nav className="border-b border-slate-800 bg-slate-950/90 backdrop-blur sticky top-0 z-40">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
        {/* Brand */}
        <Link href="/" className="flex items-center gap-2.5 group">
          <div className="w-8 h-8 rounded-lg bg-blue-600 group-hover:bg-blue-500 transition flex items-center justify-center font-bold text-white text-base shadow-lg shadow-blue-500/20">
            S
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-slate-100 tracking-wide text-sm sm:text-base">SAM-AI</span>
              <span className="text-[10px] font-semibold uppercase px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20">
                Research MVP
              </span>
            </div>
            <p className="text-[11px] text-slate-400 hidden sm:block">Parallax Intelligence Lab</p>
          </div>
        </Link>

        {/* Desktop Navigation */}
        <div className="hidden md:flex items-center gap-1">
          {navLinks.map((link) => {
            const Icon = link.icon;
            const active = isActive(link.href);
            return (
              <Link
                key={link.href}
                href={link.href}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition ${
                  active
                    ? "bg-blue-600/15 text-blue-400 border border-blue-500/20"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-900"
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                <span>{link.label}</span>
              </Link>
            );
          })}
        </div>

        {/* External Links & Actions */}
        <div className="hidden md:flex items-center gap-2.5">
          <a
            href="https://github.com/samrishtt/SAM-AI"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-800 text-xs font-medium transition"
          >
            <Github className="w-3.5 h-3.5 text-slate-400" />
            <span>GitHub</span>
            <ExternalLink className="w-3 h-3 text-slate-500" />
          </a>

          <Link
            href="/chat"
            className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-md bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold shadow-md shadow-blue-500/20 transition"
          >
            <span>Launch Chat</span>
          </Link>
        </div>

        {/* Mobile menu button */}
        <button
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          aria-label="Toggle navigation menu"
          className="md:hidden p-2 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-900 transition"
        >
          {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>
      </div>

      {/* Mobile drawer */}
      {mobileMenuOpen && (
        <div className="md:hidden border-b border-slate-800 bg-slate-950 px-4 py-3 space-y-1">
          {navLinks.map((link) => {
            const Icon = link.icon;
            const active = isActive(link.href);
            return (
              <Link
                key={link.href}
                href={link.href}
                onClick={() => setMobileMenuOpen(false)}
                className={`flex items-center gap-2.5 px-3 py-2 rounded-lg text-xs font-medium transition ${
                  active
                    ? "bg-blue-600/15 text-blue-400 border border-blue-500/20"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-900"
                }`}
              >
                <Icon className="w-4 h-4" />
                <span>{link.label}</span>
              </Link>
            );
          })}
          <div className="pt-2 border-t border-slate-900 flex flex-col gap-2">
            <a
              href="https://github.com/samrishtt/SAM-AI"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center justify-center gap-2 px-3 py-2 rounded-lg bg-slate-900 text-slate-300 border border-slate-800 text-xs font-medium"
            >
              <Github className="w-4 h-4" />
              <span>GitHub Repository</span>
            </a>
          </div>
        </div>
      )}
    </nav>
  );
};
