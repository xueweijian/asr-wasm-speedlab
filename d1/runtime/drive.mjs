// d1/runtime/drive.mjs — headless-Chrome driver for parity.html
// usage: node drive.mjs <chromePath> <url> [timeoutMs]
import puppeteer from "puppeteer-core";

const [chromePath, url, timeoutArg] = process.argv.slice(2);
const timeout = Number(timeoutArg || 1500000);

const browser = await puppeteer.launch({
  executablePath: chromePath,
  headless: true,
  args: ["--enable-unsafe-swiftshader", "--disable-dev-shm-usage",
         "--enable-features=Vulkan"],
});
try {
  const page = await browser.newPage();
  page.on("console", (m) => console.log("[page]", m.text()));
  page.on("pageerror", (e) => console.log("[pageerr]", e.message));
  await page.goto(url, { timeout: 60000 });
  try {
    await page.waitForFunction("window.__done === true", { timeout, polling: 2000 });
  } catch {
    console.log("TIMEOUT waiting for __done");
  }
  const res = await page.evaluate("window.__result");
  console.log("RESULT_JSON:" + JSON.stringify(res));
  process.exit(res && res.pass ? 0 : 3);
} finally {
  await browser.close();
}
