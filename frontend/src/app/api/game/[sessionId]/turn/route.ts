import { NextResponse } from "next/server";

import { runBridge } from "@/lib/bridge";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type RouteContext = {
  params: Promise<{
    sessionId: string;
  }>;
};

export async function POST(request: Request, { params }: RouteContext) {
  try {
    const body = await request.json().catch(() => ({}));
    const { sessionId } = await params;
    const result = await runBridge("step", body, sessionId);
    return NextResponse.json(result);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Could not play turn." },
      { status: 500 },
    );
  }
}
