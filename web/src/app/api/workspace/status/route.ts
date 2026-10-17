import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

export async function GET() {
  const baseUrl = process.env.SAM_AI_BASE_URL?.trim();
  const workspaceRoot = process.env.SAM_AI_WORKSPACE_ROOT?.trim();

  // 1. Try querying backend inference server workspace endpoint if configured
  if (baseUrl) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 2000);
      const res = await fetch(`${baseUrl.replace(/\/+$/, "")}/v1/workspace/status`, {
        signal: controller.signal,
        headers: process.env.SAM_AI_API_KEY
          ? { Authorization: `Bearer ${process.env.SAM_AI_API_KEY.trim()}` }
          : {},
      });
      clearTimeout(timeoutId);

      if (res.ok) {
        const data = await res.json();
        return NextResponse.json({
          configured: Boolean(data.configured),
          status: data.status || "ready",
          workspace_name: data.workspace_name || "workspace",
          read_only: true,
        });
      }
    } catch {
      // Backend unreachable; proceed to fallback check
    }
  }

  // 2. Direct server-side environment check if workspace_root is configured
  if (workspaceRoot) {
    const path = await import("path");
    const fs = await import("fs");
    try {
      const resolved = path.resolve(workspaceRoot);
      if (fs.existsSync(resolved) && fs.statSync(resolved).isDirectory()) {
        return NextResponse.json({
          configured: true,
          status: "ready",
          workspace_name: path.basename(resolved),
          read_only: true,
        });
      }
    } catch {
      // Fall through to unconfigured
    }
  }

  return NextResponse.json({
    configured: false,
    status: "workspace_not_configured",
    workspace_name: undefined,
    read_only: true,
    message: "No safe workspace root is configured. Set SAM_AI_WORKSPACE_ROOT or start the SAM-AI server with --workspace-root.",
  });
}
