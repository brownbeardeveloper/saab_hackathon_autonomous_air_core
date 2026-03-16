import { NextResponse } from "next/server";

import { runBridge } from "@/lib/bridge";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type RouteContext = {
  params: Promise<{
    sessionId: string;
  }>;
};

export async function GET(_request: Request, { params }: RouteContext) {
  try {
    const { sessionId } = await params;
    const result = await runBridge("state", {}, sessionId);
    return NextResponse.json(result);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Could not load game." },
      { status: 500 },
    );
  }
}
