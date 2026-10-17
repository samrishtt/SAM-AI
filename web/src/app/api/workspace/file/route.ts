import { NextRequest, NextResponse } from "next/server";
import path from "path";
import fs from "fs";

export const dynamic = "force-dynamic";

const MAX_WORKSPACE_FILE_SIZE_BYTES = 512 * 1024; // 512 KB

const DISALLOWED_DIR_NAMES = new Set([
  ".git",
  "node_modules",
  ".next",
  "__pycache__",
  ".pytest_cache",
  ".venv",
  "venv",
  "dist",
  "build",
  "target",
  ".idea",
  ".vscode",
]);

const DISALLOWED_EXTENSIONS = new Set([
  ".key",
  ".pem",
  ".safetensors",
  ".bin",
  ".pt",
  ".pth",
  ".zip",
  ".tar",
  ".gz",
  ".bz2",
  ".xz",
  ".7z",
  ".exe",
  ".dll",
  ".so",
  ".dylib",
  ".pyc",
  ".pkl",
]);

const DISALLOWED_PREFIXES = [".env", "credentials", "id_rsa", "id_ed25519", "secret"];

function isSafeRelativePath(relPath: string): boolean {
  const parts = relPath.split(/[/\\]/);
  for (const part of parts) {
    const lower = part.toLowerCase();
    if (DISALLOWED_DIR_NAMES.has(lower)) return false;
    if (DISALLOWED_PREFIXES.some((pfx) => lower.startsWith(pfx))) return false;
  }
  const baseName = path.basename(relPath).toLowerCase();
  const ext = path.extname(baseName).toLowerCase();
  if (DISALLOWED_EXTENSIONS.has(ext)) return false;
  if (DISALLOWED_PREFIXES.some((pfx) => baseName.startsWith(pfx))) return false;
  return true;
}

export async function GET(req: NextRequest) {
  const searchParams = req.nextUrl.searchParams;
  const requestedPath = searchParams.get("path");

  if (!requestedPath || requestedPath.trim().length === 0) {
    return NextResponse.json(
      { error: "Query parameter 'path' is required", code: "MISSING_PATH_PARAMETER" },
      { status: 400 }
    );
  }

  // Pre-check for traversal tokens or drive letters
  const cleanPath = requestedPath.replace(/^[/\\]+/, "");
  if (cleanPath.includes(":") || cleanPath.split(/[/\\]/).some((p) => p === ".." || p === ".")) {
    return NextResponse.json(
      { error: "Path traversal forbidden", code: "PATH_TRAVERSAL_FORBIDDEN" },
      { status: 403 }
    );
  }

  const baseUrl = process.env.SAM_AI_BASE_URL?.trim();
  const workspaceRoot = process.env.SAM_AI_WORKSPACE_ROOT?.trim();

  // 1. Try querying backend inference server
  if (baseUrl) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 2500);
      const res = await fetch(
        `${baseUrl.replace(/\/+$/, "")}/v1/workspace/file?path=${encodeURIComponent(cleanPath)}`,
        {
          signal: controller.signal,
          headers: process.env.SAM_AI_API_KEY
            ? { Authorization: `Bearer ${process.env.SAM_AI_API_KEY.trim()}` }
            : {},
        }
      );
      clearTimeout(timeoutId);

      if (res.ok) {
        const data = await res.json();
        return NextResponse.json(data);
      } else {
        const errData = await res.json().catch(() => ({}));
        return NextResponse.json(errData, { status: res.status });
      }
    } catch {
      // Backend unreachable, fallback to local workspace check
    }
  }

  // 2. Direct local file reading if configured
  if (workspaceRoot) {
    try {
      const rootResolved = path.resolve(workspaceRoot);
      const targetResolved = path.resolve(rootResolved, cleanPath);

      // Verify containment within root
      const rel = path.relative(rootResolved, targetResolved);
      if (rel.startsWith("..") || path.isAbsolute(rel)) {
        return NextResponse.json(
          { error: "Path traversal forbidden", code: "PATH_TRAVERSAL_FORBIDDEN" },
          { status: 403 }
        );
      }

      // Check real symlink path
      const realTarget = fs.realpathSync(targetResolved);
      const realRoot = fs.realpathSync(rootResolved);
      if (!realTarget.startsWith(realRoot)) {
        return NextResponse.json(
          { error: "Symlink escape forbidden", code: "SYMLINK_ESCAPE_FORBIDDEN" },
          { status: 403 }
        );
      }

      // Check sensitive paths
      if (!isSafeRelativePath(rel)) {
        return NextResponse.json(
          { error: "Access to sensitive file forbidden", code: "SENSITIVE_FILE_FORBIDDEN" },
          { status: 403 }
        );
      }

      if (!fs.existsSync(targetResolved) || !fs.statSync(targetResolved).isFile()) {
        return NextResponse.json(
          { error: `File not found: ${rel.replace(/\\/g, "/")}`, code: "FILE_NOT_FOUND" },
          { status: 404 }
        );
      }

      const stat = fs.statSync(targetResolved);
      if (stat.size > MAX_WORKSPACE_FILE_SIZE_BYTES) {
        return NextResponse.json(
          { error: "File exceeds 512KB size limit", code: "FILE_TOO_LARGE" },
          { status: 413 }
        );
      }

      const content = fs.readFileSync(targetResolved, "utf-8");
      return NextResponse.json({
        path: rel.replace(/\\/g, "/"),
        name: path.basename(targetResolved),
        content,
        size_bytes: stat.size,
        read_only: true,
      });
    } catch (err: any) {
      if (err.code === "ENOENT") {
        return NextResponse.json(
          { error: "File not found", code: "FILE_NOT_FOUND" },
          { status: 404 }
        );
      }
      return NextResponse.json(
        { error: `Failed to read file: ${err.message}`, code: "FILE_READ_ERROR" },
        { status: 400 }
      );
    }
  }

  return NextResponse.json(
    { error: "Workspace root is not configured", code: "WORKSPACE_NOT_CONFIGURED" },
    { status: 400 }
  );
}
