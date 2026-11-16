import { connection } from "next/server";

import { readCameraPresets } from "@/lib/cameras";

export async function GET() {
  await connection(); // nie vorrendern, Datei kann sich jederzeit aendern
  return Response.json({ presets: readCameraPresets() });
}
