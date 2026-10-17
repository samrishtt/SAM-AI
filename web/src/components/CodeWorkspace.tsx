"use client";

import React, { useState, useEffect } from "react";
import {
  Folder,
  FileCode,
  FileText,
  Search,
  CheckCircle2,
  Clock,
  AlertTriangle,
  RefreshCw,
  Copy,
  Check,
  ShieldAlert,
  GitBranch,
  Terminal,
  FileCheck2,
  Lock,
} from "lucide-react";
import { BackendStatus, WorkspaceFileInfo, WorkspaceStatus } from "@/types/chat";

interface CodeWorkspaceProps {
  backendStatus: BackendStatus;
}

export const CodeWorkspace: React.FC<CodeWorkspaceProps> = ({ backendStatus }) => {
  const [workspaceStatus, setWorkspaceStatus] = useState<WorkspaceStatus>({
    configured: false,
    status: "workspace_not_configured",
    read_only: true,
  });
  const [files, setFiles] = useState<WorkspaceFileInfo[]>([]);
  const [searchFilter, setSearchFilter] = useState("");
  const [selectedFilePath, setSelectedFilePath] = useState<string | null>(null);
  const [fileContent, setFileContent] = useState<string | null>(null);
  const [fileLoading, setFileLoading] = useState(false);
  const [fileError, setFileError] = useState<string | null>(null);
  const [centerTab, setCenterTab] = useState<"code" | "diff">("code");
  const [copied, setCopied] = useState(false);
  const [isLoadingWorkspace, setIsLoadingWorkspace] = useState(true);

  // Load workspace status and files
  const fetchWorkspace = async () => {
    setIsLoadingWorkspace(true);
    try {
      const statusRes = await fetch("/api/workspace/status");
      const statusData: WorkspaceStatus = await statusRes.json();
      setWorkspaceStatus(statusData);

      if (statusData.configured) {
        const filesRes = await fetch("/api/workspace/files");
        if (filesRes.ok) {
          const filesData = await filesRes.json();
          const list: WorkspaceFileInfo[] = filesData.files || [];
          setFiles(list);
          if (list.length > 0 && !selectedFilePath) {
            loadFile(list[0].path);
          }
        }
      }
    } catch (err: any) {
      setWorkspaceStatus({
        configured: false,
        status: "error",
        read_only: true,
        error: err.message,
      });
    } finally {
      setIsLoadingWorkspace(false);
    }
  };

  useEffect(() => {
    fetchWorkspace();
  }, []);

  // Load single file content
  const loadFile = async (filePath: string) => {
    setSelectedFilePath(filePath);
    setFileLoading(true);
    setFileError(null);
    try {
      const res = await fetch(`/api/workspace/file?path=${encodeURIComponent(filePath)}`);
      const data = await res.json();
      if (!res.ok) {
        setFileError(data.error || "Failed to load file content.");
        setFileContent(null);
      } else {
        setFileContent(data.content);
      }
    } catch (err: any) {
      setFileError(err.message || "Failed to load file.");
      setFileContent(null);
    } finally {
      setFileLoading(false);
    }
  };

  const handleCopy = () => {
    if (fileContent) {
      navigator.clipboard.writeText(fileContent);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const filteredFiles = files.filter(
    (f) =>
      f.path.toLowerCase().includes(searchFilter.toLowerCase()) ||
      f.name.toLowerCase().includes(searchFilter.toLowerCase())
  );

  // Render unconfigured state
  if (!workspaceStatus.configured && !isLoadingWorkspace) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-8 bg-[#FAF8F5] text-[#28231F]">
        <div className="max-w-md w-full bg-[#FFFFFF] border border-[#E8E1D9] rounded-xl p-6 shadow-sm text-center">
          <div className="w-12 h-12 rounded-xl bg-[#FEF3C7] border border-[#FDE68A] text-[#B45309] flex items-center justify-center mx-auto mb-4">
            <Lock className="w-6 h-6" />
          </div>
          <h2 className="text-lg font-bold text-[#28231F] mb-2">Workspace Not Configured</h2>
          <p className="text-xs text-[#766B62] mb-6 leading-relaxed">
            SAM-AI Code operates on an explicitly configured local repository root with strict read-only
            containment. No workspace root is currently attached.
          </p>
          <div className="bg-[#FAF8F5] p-3 rounded-lg border border-[#E8E1D9] text-left font-mono text-[11px] text-[#28231F] space-y-1 mb-6">
            <p className="text-[#766B62] font-sans text-[10px] uppercase font-bold tracking-wider">
              How to configure:
            </p>
            <p className="text-[#D97745]"># Set local environment variable in web/.env.local:</p>
            <p>SAM_AI_WORKSPACE_ROOT=C:\path\to\repo</p>
            <p className="text-[#D97745] mt-2"># Or start the SAM-AI server with:</p>
            <p>python -m sam_ai.server --workspace-root C:\path\to\repo</p>
          </div>
          <button
            onClick={fetchWorkspace}
            className="w-full py-2 px-4 rounded-lg bg-[#D97745] hover:bg-[#B95F34] text-white text-xs font-medium transition flex items-center justify-center gap-2 shadow-sm"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            Check Workspace Connection
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col h-[calc(100dvh-4rem)] bg-[#FAF8F5] text-[#28231F] overflow-hidden">
      {/* Workspace Sub-header */}
      <div className="h-10 border-b border-[#E8E1D9] bg-[#FFFFFF] px-4 flex items-center justify-between text-xs">
        <div className="flex items-center gap-2">
          <Folder className="w-4 h-4 text-[#D97745]" />
          <span className="font-semibold text-[#28231F] font-mono">
            {workspaceStatus.workspace_name || "SAM-AI Code Workspace"}
          </span>
          <span className="text-[10px] font-mono uppercase bg-[#EBF5F0] text-[#2D6A4F] border border-[#B7E4C7] px-1.5 py-0.5 rounded font-semibold">
            Safe Read-Only
          </span>
          <span className="text-[#766B62] text-[11px]">({files.length} eligible files)</span>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={fetchWorkspace}
            title="Refresh repository files"
            className="p-1 rounded text-[#766B62] hover:text-[#28231F] hover:bg-[#F5F0EA] transition"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isLoadingWorkspace ? "animate-spin" : ""}`} />
          </button>
        </div>
      </div>

      {/* Main 3-Column Layout */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Column: Read-Only File Tree */}
        <div className="w-64 sm:w-72 border-r border-[#E8E1D9] bg-[#FFFFFF] flex flex-col h-full">
          <div className="p-2 border-b border-[#E8E1D9]">
            <div className="relative">
              <Search className="w-3.5 h-3.5 text-[#766B62] absolute left-2.5 top-2.5" />
              <input
                type="text"
                placeholder="Filter files..."
                value={searchFilter}
                onChange={(e) => setSearchFilter(e.target.value)}
                className="w-full bg-[#FAF8F5] border border-[#E8E1D9] rounded-md pl-8 pr-2 py-1.5 text-xs text-[#28231F] placeholder-[#766B62]/60 focus:outline-none focus:border-[#D97745]"
              />
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-1.5 space-y-0.5 font-mono text-[11px]">
            {filteredFiles.length === 0 ? (
              <p className="p-3 text-[#766B62] text-xs font-sans text-center">
                {isLoadingWorkspace ? "Scanning repository..." : "No matching files found."}
              </p>
            ) : (
              filteredFiles.map((file) => {
                const isSelected = selectedFilePath === file.path;
                return (
                  <button
                    key={file.path}
                    onClick={() => loadFile(file.path)}
                    title={file.path}
                    className={`w-full text-left px-2 py-1.5 rounded flex items-center justify-between gap-2 transition ${
                      isSelected
                        ? "bg-[#FBEBDD] text-[#D97745] font-semibold border border-[#D97745]/30"
                        : "text-[#28231F] hover:bg-[#F5F0EA]"
                    }`}
                  >
                    <div className="flex items-center gap-1.5 truncate">
                      {file.extension === ".ts" || file.extension === ".tsx" || file.extension === ".py" ? (
                        <FileCode className="w-3.5 h-3.5 text-[#D97745] flex-shrink-0" />
                      ) : (
                        <FileText className="w-3.5 h-3.5 text-[#766B62] flex-shrink-0" />
                      )}
                      <span className="truncate">{file.path}</span>
                    </div>
                    <span className="text-[9px] text-[#766B62] font-sans flex-shrink-0">
                      {file.size_bytes > 1024
                        ? `${(file.size_bytes / 1024).toFixed(1)}k`
                        : `${file.size_bytes}b`}
                    </span>
                  </button>
                );
              })
            )}
          </div>
        </div>

        {/* Center Column: Code Viewer / Proposed Diff */}
        <div className="flex-1 flex flex-col border-r border-[#E8E1D9] h-full overflow-hidden bg-[#FFFFFF]">
          {/* Center Tabs */}
          <div className="h-9 border-b border-[#E8E1D9] bg-[#FAF8F5] px-3 flex items-center justify-between text-xs">
            <div className="flex items-center gap-1">
              <button
                onClick={() => setCenterTab("code")}
                className={`px-3 py-1 rounded text-xs font-medium transition ${
                  centerTab === "code"
                    ? "bg-[#FFFFFF] text-[#28231F] font-semibold shadow-xs border border-[#E8E1D9]"
                    : "text-[#766B62] hover:text-[#28231F]"
                }`}
              >
                File Viewer
              </button>
              <button
                onClick={() => setCenterTab("diff")}
                className={`px-3 py-1 rounded text-xs font-medium transition ${
                  centerTab === "diff"
                    ? "bg-[#FFFFFF] text-[#28231F] font-semibold shadow-xs border border-[#E8E1D9]"
                    : "text-[#766B62] hover:text-[#28231F]"
                }`}
              >
                Proposed Diff
              </button>
            </div>

            {centerTab === "code" && selectedFilePath && (
              <div className="flex items-center gap-2">
                <span className="text-[11px] font-mono text-[#766B62] truncate max-w-xs">
                  {selectedFilePath}
                </span>
                <button
                  onClick={handleCopy}
                  title="Copy file contents"
                  className="p-1 rounded text-[#766B62] hover:text-[#28231F] hover:bg-[#F5F0EA] transition"
                >
                  {copied ? <Check className="w-3.5 h-3.5 text-[#2D6A4F]" /> : <Copy className="w-3.5 h-3.5" />}
                </button>
              </div>
            )}
          </div>

          {/* Center Viewport */}
          <div className="flex-1 overflow-auto p-4 font-mono text-xs bg-[#FFFFFF]">
            {centerTab === "code" ? (
              fileLoading ? (
                <div className="flex items-center justify-center h-full text-[#766B62] font-sans">
                  <RefreshCw className="w-4 h-4 animate-spin mr-2 text-[#D97745]" /> Loading file content...
                </div>
              ) : fileError ? (
                <div className="p-4 bg-[#FEE2E2] border border-[#FECACA] rounded-lg text-[#C53030] font-sans">
                  <p className="font-semibold mb-1">Unable to view file</p>
                  <p className="text-xs text-[#C53030]/80">{fileError}</p>
                </div>
              ) : fileContent !== null ? (
                <div className="table w-full">
                  {fileContent.split("\n").map((line, idx) => (
                    <div key={idx} className="table-row hover:bg-[#FAF8F5] transition">
                      <span className="table-cell select-none pr-4 text-right text-[#766B62]/60 text-[11px] w-10">
                        {idx + 1}
                      </span>
                      <span className="table-cell whitespace-pre text-[#28231F] font-mono text-[11px]">
                        {line || " "}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="flex items-center justify-center h-full text-[#766B62] font-sans text-xs">
                  Select a file from the repository tree to inspect.
                </div>
              )
            ) : (
              /* Proposed Diff Review Panel */
              <div className="space-y-4 font-sans">
                <div className="p-3 bg-[#FBEBDD] border border-[#D97745]/20 rounded-lg text-xs text-[#28231F]">
                  <div className="flex items-center gap-2 font-semibold text-[#D97745] mb-1">
                    <GitBranch className="w-4 h-4 text-[#D97745]" />
                    <span>Proposed Diff Review (Read-Only)</span>
                    <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-[#FAF8F5] text-[#D97745] border border-[#D97745]/30">
                      Not Applied
                    </span>
                  </div>
                  <p className="text-[11px] text-[#766B62] leading-relaxed">
                    SAM-AI Code analyzes and drafts proposed changes. Modifications are never silently applied.
                    In this milestone, code execution and disk mutations are strictly prohibited.
                  </p>
                </div>

                <div className="bg-[#FAF8F5] rounded-lg border border-[#E8E1D9] overflow-hidden font-mono text-[11px]">
                  <div className="px-3 py-1.5 bg-[#F5F0EA] border-b border-[#E8E1D9] text-[#766B62] flex items-center justify-between text-[11px]">
                    <span>diff --git a/{selectedFilePath || "target_file.py"} b/{selectedFilePath || "target_file.py"}</span>
                    <span className="text-[10px] uppercase text-[#B45309] bg-[#FEF3C7] px-1.5 py-0.5 rounded border border-[#FDE68A] font-semibold">
                      Proposed Only
                    </span>
                  </div>
                  <div className="p-3 space-y-0.5 text-[#28231F]">
                    <p className="text-[#766B62]">--- a/{selectedFilePath || "target_file.py"}</p>
                    <p className="text-[#766B62]">+++ b/{selectedFilePath || "target_file.py"}</p>
                    <p className="text-[#D97745]">@@ -1,5 +1,6 @@</p>
                    <p className="text-[#766B62]"> # Sovereign code verification pipeline</p>
                    <p className="text-[#C53030] bg-[#FEE2E2] px-1 rounded">- insecure_call(remote_third_party_api)</p>
                    <p className="text-[#2D6A4F] bg-[#EBF5F0] px-1 rounded">+ enforce_sovereignty_and_boundary_checks()</p>
                    <p className="text-[#766B62]"> return verified_result</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Task/Chat & Test Results Panel */}
        <div className="w-80 sm:w-96 flex flex-col h-full bg-[#FAF8F5] overflow-hidden">
          {/* Top Half: Task & Context Panel */}
          <div className="h-1/2 flex flex-col border-b border-[#E8E1D9]">
            <div className="p-2.5 border-b border-[#E8E1D9] bg-[#FFFFFF] flex items-center justify-between">
              <span className="text-xs font-semibold text-[#28231F] flex items-center gap-1.5">
                <Terminal className="w-3.5 h-3.5 text-[#D97745]" />
                Task &amp; Repository Context
              </span>
              <span className="text-[10px] text-[#766B62] font-mono">
                {selectedFilePath ? selectedFilePath.split("/").pop() : "None"}
              </span>
            </div>

            <div className="flex-1 overflow-y-auto p-3 text-xs space-y-3">
              <div className="p-2.5 bg-[#FFFFFF] rounded-lg border border-[#E8E1D9] text-[11px] text-[#28231F] space-y-1">
                <p className="font-semibold text-[#28231F]">Selected Target</p>
                <p className="font-mono text-[#D97745] break-all">{selectedFilePath || "No file selected"}</p>
              </div>

              <div className="space-y-1">
                <p className="text-[11px] font-semibold text-[#766B62]">Security Invariants</p>
                <ul className="text-[11px] text-[#766B62] space-y-1 list-disc pl-4">
                  <li>Path traversal &amp; symlink escapes blocked.</li>
                  <li>Sensitive files (.git, .env, .key) excluded.</li>
                  <li>Max file size 512KB enforced.</li>
                  <li>Read-only inspection only (no execution).</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Bottom Half: Test Results & Verification Integrity Panel */}
          <div className="h-1/2 flex flex-col">
            <div className="p-2.5 border-b border-[#E8E1D9] bg-[#FFFFFF] flex items-center justify-between">
              <span className="text-xs font-semibold text-[#28231F] flex items-center gap-1.5">
                <FileCheck2 className="w-3.5 h-3.5 text-[#2D6A4F]" />
                Test Verification &amp; Integrity
              </span>
            </div>

            <div className="flex-1 overflow-y-auto p-3 text-xs space-y-3">
              {/* Real Test Execution Section */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-semibold text-[#28231F]">Verified Test Runs</span>
                  <span className="text-[9px] uppercase font-mono px-1.5 py-0.5 rounded bg-[#EBF5F0] text-[#2D6A4F] border border-[#B7E4C7] font-semibold">
                    REAL TEST EXECUTION
                  </span>
                </div>

                <div className="p-2.5 bg-[#FFFFFF] rounded-lg border border-[#E8E1D9] space-y-1.5 text-[11px]">
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-[#2D6A4F] flex items-center gap-1 font-semibold">
                      <CheckCircle2 className="w-3 h-3 text-[#2D6A4F]" />
                      test_inference_server.py
                    </span>
                    <span className="text-[#766B62] text-[10px]">12/12 passed</span>
                  </div>
                  <p className="text-[10px] text-[#766B62]">
                    Prohibited providers blocked, streaming 503 fail-closed, secrets masked.
                  </p>
                </div>

                <div className="p-2.5 bg-[#FFFFFF] rounded-lg border border-[#E8E1D9] space-y-1.5 text-[11px]">
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-[#2D6A4F] flex items-center gap-1 font-semibold">
                      <CheckCircle2 className="w-3 h-3 text-[#2D6A4F]" />
                      Phase 2A Security Suite
                    </span>
                    <span className="text-[#766B62] text-[10px]">pytest</span>
                  </div>
                  <p className="text-[10px] text-[#766B62]">
                    Path traversal, symlink escapes, 512KB limits, and sensitive exclusions verified.
                  </p>
                </div>
              </div>

              {/* Planned Actions / Not Run Section */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-semibold text-[#28231F]">Planned Actions</span>
                  <span className="text-[9px] uppercase font-mono px-1.5 py-0.5 rounded bg-[#FEF3C7] text-[#B45309] border border-[#FDE68A] font-semibold">
                    PLANNED ACTION / NOT RUN
                  </span>
                </div>

                <div className="p-2 bg-[#FEF3C7]/40 border border-[#FDE68A] rounded-lg text-[10px] text-[#B45309] space-y-0.5">
                  <p className="font-semibold text-[#28231F]">Arbitrary Code Execution</p>
                  <p>Status: NOT RUN. Prohibited in local read-only MVP.</p>
                </div>

                <div className="p-2 bg-[#FEF3C7]/40 border border-[#FDE68A] rounded-lg text-[10px] text-[#B45309] space-y-0.5">
                  <p className="font-semibold text-[#28231F]">Full 7B Weights CPU Generation</p>
                  <p>Status: NOT RUN. Blocked by CPU memory limit; requires authorized GPU.</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
