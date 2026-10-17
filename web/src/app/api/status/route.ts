import { NextResponse } from "next/server";
import { inspectBackendReadiness } from "@/lib/model-adapter";

export const dynamic = "force-dynamic";

export async function GET() {
  const statusInfo = await inspectBackendReadiness();
  return NextResponse.json(statusInfo);
}
