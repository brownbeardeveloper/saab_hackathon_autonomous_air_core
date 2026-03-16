import { NextResponse } from "next/server";

import { runBridge } from "@/lib/bridge";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  try {
    const body = await request.json().catch(() => ({}));
    const result = await runBridge("start", body);
    return NextResponse.json(result);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Could not start game." },
      { status: 500 },
    );
  }
}
