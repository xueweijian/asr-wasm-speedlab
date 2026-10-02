// d1/runtime/smoke.js — CPU-only structural smoke: expand all kernels without GPU
// node smoke.js  (run from the dir holding kernels.json + weights.bin)
import { Runtime } from "./rt.js";
import { readFileSync } from "node:fs";

const DIR = process.argv[2] || ".";
const doc = JSON.parse(readFileSync(DIR + "/kernels.json", "utf8"));
const wBytes = readFileSync(DIR + "/weights.bin");
const rt = new Runtime(doc, wBytes);
rt.layout();
console.log(`arena A: ${(rt.aU32 * 4 / 1e6).toFixed(1)}MB  tensors=${rt.A.size}`);
rt.expand();
const nDisp = rt.groups.reduce((s, g) => s + g.dispatches.length, 0);
const byPipe = {};
for (const g of rt.groups) for (const d of g.dispatches) byPipe[d.pipe] = (byPipe[d.pipe] || 0) + 1;
console.log(`groups=${rt.groups.length} dispatches=${nDisp}`);
console.log("by pipe:", JSON.stringify(byPipe));
console.log("SMOKE PASS");
