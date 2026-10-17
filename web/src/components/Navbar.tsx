"use client";

import React, { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { MessageSquare, BookOpen, Activity, Github, Menu, X, ArrowUpRight } from "lucide-react";

export const Navbar: React.FC = () => {
  const pathname = usePathname();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navLinks = [
    { href: "/chat", label: "Chat", icon: MessageSquare },
    { href: "/research", label: "Research", icon: BookOpen },
    { href: "/status", label: "Status", icon: Activity },
  ];

  const isActive = (href: string) => {
    if (href === "/") return pathname === "/";
    return pathname.startsWith(href);
  };

  return (
    <nav className="border-b border-[#E8E1D9] bg-[#FAF8F5]/90 backdrop-blur-md sticky top-0 z-40">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
        {/* Brand */}
        <Link href="/" className="flex items-center gap-2.5 group">
          <div className="w-8 h-8 rounded-lg bg-[#D97745] group-hover:bg-[#B95F34] transition flex items-center justify-center font-serif font-bold text-white text-base shadow-sm">
            S
          </div>
          <div className="flex items-baseline gap-2">
            <span className="font-semibold text-[#28231F] tracking-tight text-base sm:text-lg">SAM-AI</span>
            <span className="text-[11px] text-[#766B62] font-normal hidden sm:inline">
              Parallax Intelligence Lab
            </span>
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
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition ${
                  active
                    ? "bg-[#FBEBDD] text-[#D97745]"
                    : "text-[#766B62] hover:text-[#28231F] hover:bg-[#F5F0EA]"
                }`}
              >
                <Icon className="w-4 h-4" />
                <span>{link.label}</span>
              </Link>
            );
          })}
        </div>

        {/* Right Action */}
        <div className="hidden md:flex items-center gap-3">
          <a
            href="https://github.com/samrishtt/SAM-AI"
            target="_blank"
            rel="noopener noreferrer"
            title="GitHub Repository"
            className="p-2 rounded-lg text-[#766B62] hover:text-[#28231F] hover:bg-[#F5F0EA] transition"
          >
            <Github className="w-4 h-4" />
          </a>
          <Link
            href="/chat"
            className="flex items-center gap-1 px-3.5 py-1.5 rounded-lg text-xs font-medium bg-[#D97745] hover:bg-[#B95F34] text-white shadow-sm transition"
          >
            <span>Open Chat</span>
            <ArrowUpRight className="w-3.5 h-3.5 opacity-80" />
          </Link>
        </div>

        {/* Mobile Menu Button */}
        <div className="flex items-center gap-2 md:hidden">
          <Link
            href="/chat"
            className="px-3 py-1 rounded-lg text-xs font-medium bg-[#D97745] text-white"
          >
            Chat
          </Link>
          <button
            onClick={() => setMobileMenuOpen((prev) => !prev)}
            aria-label="Toggle Navigation"
            className="p-2 rounded-lg text-[#766B62] hover:text-[#28231F] hover:bg-[#F5F0EA] transition"
          >
            {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>
      </div>

      {/* Mobile Menu Dropdown */}
      {mobileMenuOpen && (
        <div className="md:hidden border-b border-[#E8E1D9] bg-[#FFFFFF] px-4 py-3 space-y-1 shadow-sm">
          <Link
            href="/"
            onClick={() => setMobileMenuOpen(false)}
            className={`block px-3 py-2 rounded-lg text-sm font-medium ${
              pathname === "/"
                ? "bg-[#FBEBDD] text-[#D97745]"
                : "text-[#766B62] hover:text-[#28231F] hover:bg-[#F5F0EA]"
            }`}
          >
            Overview
          </Link>
          {navLinks.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              onClick={() => setMobileMenuOpen(false)}
              className={`block px-3 py-2 rounded-lg text-sm font-medium ${
                isActive(link.href)
                  ? "bg-[#FBEBDD] text-[#D97745]"
                  : "text-[#766B62] hover:text-[#28231F] hover:bg-[#F5F0EA]"
              }`}
            >
              {link.label}
            </Link>
          ))}
          <a
            href="https://github.com/samrishtt/SAM-AI"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center justify-between px-3 py-2 rounded-lg text-sm font-medium text-[#766B62] hover:text-[#28231F] hover:bg-[#F5F0EA]"
          >
            <span>GitHub Repository</span>
            <Github className="w-4 h-4" />
          </a>
        </div>
      )}
    </nav>
  );
};
