"use client";

import React, { useState } from "react";
import { Lock, KeyRound, Check, X } from "lucide-react";
import { setStoredPassword, getStoredPassword, clearStoredPassword } from "@/lib/storage";

interface PasswordModalProps {
  isOpen: boolean;
  onClose: () => void;
  onPasswordSaved: () => void;
}

export const PasswordModal: React.FC<PasswordModalProps> = ({
  isOpen,
  onClose,
  onPasswordSaved,
}) => {
  const [passwordInput, setPasswordInput] = useState(getStoredPassword());
  const [statusMsg, setStatusMsg] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleSave = (e: React.FormEvent) => {
    e.preventDefault();
    const clean = passwordInput.trim();
    if (!clean) {
      clearStoredPassword();
      setStatusMsg("Password cleared. You are currently unauthorized if the gate is active.");
      onPasswordSaved();
      return;
    }

    setStoredPassword(clean);
    setStatusMsg("Access password saved locally.");
    setTimeout(() => {
      onPasswordSaved();
      onClose();
    }, 600);
  };

  const handleClear = () => {
    clearStoredPassword();
    setPasswordInput("");
    setStatusMsg("Stored password removed.");
    onPasswordSaved();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm p-4">
      <div className="bg-[#FFFFFF] border border-[#E8E1D9] rounded-xl max-w-md w-full p-6 shadow-xl relative text-[#28231F]">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-[#766B62] hover:text-[#28231F] p-1 rounded-lg hover:bg-[#F5F0EA] transition"
        >
          <X className="w-5 h-5" />
        </button>

        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-lg bg-[#FBEBDD] border border-[#D97745]/20 text-[#D97745] flex items-center justify-center">
            <KeyRound className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-semibold text-[#28231F]">Site Access Gate</h3>
            <p className="text-xs text-[#766B62]">Restricts API quota consumption</p>
          </div>
        </div>

        <p className="text-xs text-[#766B62] leading-relaxed mb-4">
          If this SAM-AI instance is configured with <code className="text-[#D97745] bg-[#F5F0EA] px-1 py-0.5 rounded font-mono text-[11px]">SAM_AI_SITE_PASSWORD</code>,
          you must provide the matching password below. The password is kept in your browser and sent with requests via the{" "}
          <code className="text-[#D97745] bg-[#F5F0EA] px-1 py-0.5 rounded font-mono text-[11px]">x-sam-ai-password</code> header.
        </p>

        <form onSubmit={handleSave} className="space-y-4">
          <div>
            <label className="block text-xs font-medium text-[#28231F] mb-1.5">
              Access Password
            </label>
            <div className="relative">
              <input
                type="password"
                value={passwordInput}
                onChange={(e) => setPasswordInput(e.target.value)}
                placeholder="Enter SAM_AI_SITE_PASSWORD..."
                className="w-full bg-[#FAF8F5] border border-[#E8E1D9] rounded-lg px-3.5 py-2.5 text-sm text-[#28231F] placeholder:text-[#766B62]/60 focus:outline-none focus:border-[#D97745] transition"
              />
            </div>
          </div>

          {statusMsg && (
            <p className="text-xs text-[#2D6A4F] bg-[#EBF5F0] border border-[#B7E4C7] px-3 py-2 rounded-lg flex items-center gap-2">
              <Check className="w-3.5 h-3.5" />
              {statusMsg}
            </p>
          )}

          <div className="flex items-center justify-between pt-2">
            <button
              type="button"
              onClick={handleClear}
              className="text-xs text-[#766B62] hover:text-[#C53030] transition"
            >
              Clear Password
            </button>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={onClose}
                className="px-3.5 py-2 rounded-lg text-xs font-medium text-[#766B62] hover:bg-[#F5F0EA] hover:text-[#28231F] transition"
              >
                Close
              </button>
              <button
                type="submit"
                className="px-4 py-2 rounded-lg text-xs font-medium bg-[#D97745] hover:bg-[#B95F34] text-white transition shadow-sm"
              >
                Save &amp; Unlock
              </button>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
};
