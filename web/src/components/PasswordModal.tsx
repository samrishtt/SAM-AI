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
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
      <div className="bg-slate-900 border border-slate-800 rounded-xl max-w-md w-full p-6 shadow-2xl relative">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-slate-400 hover:text-white p-1 rounded-lg hover:bg-slate-800"
        >
          <X className="w-5 h-5" />
        </button>

        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-lg bg-blue-600/10 border border-blue-500/20 text-blue-400 flex items-center justify-center">
            <KeyRound className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-semibold text-slate-100">Site Access Gate</h3>
            <p className="text-xs text-slate-400">Restricts API quota consumption</p>
          </div>
        </div>

        <p className="text-xs text-slate-300 leading-relaxed mb-4">
          If this SAM-AI instance is configured with <code className="text-blue-300 bg-slate-800 px-1 py-0.5 rounded">SAM_AI_SITE_PASSWORD</code>,
          you must provide the matching password below. The password is kept in your browser and sent with requests via the{" "}
          <code className="text-blue-300 bg-slate-800 px-1 py-0.5 rounded">x-sam-ai-password</code> header.
        </p>

        <form onSubmit={handleSave} className="space-y-4">
          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1.5">
              Access Password
            </label>
            <div className="relative">
              <input
                type="password"
                value={passwordInput}
                onChange={(e) => setPasswordInput(e.target.value)}
                placeholder="Enter SAM_AI_SITE_PASSWORD..."
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3.5 py-2.5 text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none focus:border-blue-500 transition"
              />
            </div>
          </div>

          {statusMsg && (
            <p className="text-xs text-blue-400 bg-blue-500/10 border border-blue-500/20 px-3 py-2 rounded-lg flex items-center gap-2">
              <Check className="w-3.5 h-3.5" />
              {statusMsg}
            </p>
          )}

          <div className="flex items-center justify-between pt-2">
            <button
              type="button"
              onClick={handleClear}
              className="text-xs text-slate-400 hover:text-rose-400 transition"
            >
              Clear Password
            </button>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={onClose}
                className="px-3.5 py-2 rounded-lg text-xs font-medium text-slate-300 hover:bg-slate-800 transition"
              >
                Close
              </button>
              <button
                type="submit"
                className="px-4 py-2 rounded-lg text-xs font-medium bg-blue-600 hover:bg-blue-500 text-white transition shadow-md shadow-blue-500/20"
              >
                Save & Unlock
              </button>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
};
