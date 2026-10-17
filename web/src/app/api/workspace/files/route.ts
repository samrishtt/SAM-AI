import { NextResponse } from "next/server";
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

function scanDirectory(dir: string, rootDir: string, maxFiles = 1000): Array<{ path: string; name: string; size_bytes: number; extension: string }> {
  const results: Array<{ path: string; name: string; size_bytes: number; extension: string }> = [];
  const entries = fs.readdirSync(dir, { withFileTypes: true });

  for (const entry of entries) {
    if (results.length >= maxFiles) break;

    const fullPath = path.join(dir, entry.name);
    const relPath = path.relative(rootDir, fullPath);

    // Symlink escape check
    try {
      const real = fs.realpathSync(fullPath);
      const realRoot = fs.realpathSync(rootDir);
      if (!real.startsWith(realRoot)) continue;
    } catch {
      continue;
    }

    if (!isSafeRelativePath(relPath)) continue;

    if (entry.isDirectory()) {
      results.push(...scanDirectory(fullPath, rootDir, maxFiles - results.length));
    } else if (entry.isFile()) {
      try {
        const stat = fs.statSync(fullPath);
        if (stat.size <= MAX_WORKSPACE_FILE_SIZE_BYTES) {
          results.push({
            path: relPath.replace(/\\/g, "/"),
            name: entry.name,
            size_bytes: stat.size,
            extension: path.extname(entry.name).toLowerCase(),
          });
        }
      } catch {
        continue;
      }
    }
  }

  return results;
}

export async function GET() {
  const baseUrl = process.env.SAM_AI_BASE_URL?.trim();
  const workspaceRoot = process.env.SAM_AI_WORKSPACE_ROOT?.trim();

  // 1. Try querying backend inference server
  if (baseUrl) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 2500);
      const res = await fetch(`${baseUrl.replace(/\/+$/, "")}/v1/workspace/files`, {
        signal: controller.signal,
        headers: process.env.SAM_AI_API_KEY
          ? { Authorization: `Bearer ${process.env.SAM_AI_API_KEY.trim()}` }
          : {},
      });
      clearTimeout(timeoutId);

      if (res.ok) {
        const data = await res.json();
        return NextResponse.json(data);
      }
    } catch {
      // Backend unreachable, fallback to local workspace if configured
    }
  }

  // 2. Direct local scanning if configured
  if (workspaceRoot) {
    try {
      const resolved = path.resolve(workspaceRoot);
      if (fs.existsSync(resolved) && fs.statSync(resolved).isDirectory()) {
        const files = scanDirectory(resolved, resolved);
        files.sort((a, b) => a.path.localeCompare(b.path));
        return NextResponse.json({
          workspace_name: path.basename(resolved),
          count: files.length,
          files,
          read_only: true,
        });
      }
    } catch (err: any) {
      return NextResponse.json(
        { error: `Failed to inspect workspace: ${err.message}`, code: "WORKSPACE_INSPECTION_FAILED" },
        { status: 500 }
      );
    }
  }

  return NextResponse.json(
    { error: "Workspace root is not configured", code: "WORKSPACE_NOT_CONFIGURED" },
    { status: 400 }
  );
}
