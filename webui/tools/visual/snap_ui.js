"use strict";
const fs   = require("fs");
const os   = require("os");
const path = require("path");

class UiSnapper {

  DEFAULT_ROUTES = ["home", "model", "pipelines", "architectures", "scripts", "configuration", "console"];

  constructor(argv) {
    this.port    = 8765;
    this.outDir  = "/tmp/tomosar-ui";
    this.settle  = 1800;
    this.width   = 1600;
    this.height  = 950;
    this.routes  = [];
    this.parse(argv);
    if (!this.routes.length) this.routes = this.DEFAULT_ROUTES;
  }

  parse(argv) {
    for (let i = 0; i < argv.length; i++) {
      const a = argv[i];
      if      (a === "--port")   this.port   = Number(argv[++i]);
      else if (a === "--out")    this.outDir = path.resolve(argv[++i]);
      else if (a === "--settle") this.settle = Number(argv[++i]);
      else if (a === "--width")  this.width  = Number(argv[++i]);
      else if (a === "--height") this.height = Number(argv[++i]);
      else if (a.startsWith("--")) throw new Error("unknown flag: " + a);
      else this.routes.push(a);
    }
  }

  resolveChromium() {
    if (process.env.PW_CHROMIUM) return process.env.PW_CHROMIUM;

    const root = path.join(os.homedir(), ".cache", "ms-playwright");
    const dirs = fs.readdirSync(root).filter(d => d.startsWith("chromium")).sort().reverse();
    for (const d of dirs) {
      for (const rel of ["chrome-headless-shell-linux64/chrome-headless-shell", "chrome-linux/chrome"]) {
        const exe = path.join(root, d, rel);
        if (fs.existsSync(exe)) return exe;
      }
    }
    throw new Error("no chromium found under " + root + " (set PW_CHROMIUM)");
  }

  async snapRoute(page, route) {
    await page.evaluate((r) => { window.location.hash = "#/" + r; }, route);
    await page.waitForLoadState("networkidle");
    await page.evaluate(() => document.fonts.ready);
    await page.waitForTimeout(this.settle);

    const name = route.replace(/\//g, "_") + ".png";
    const file = path.join(this.outDir, name);
    await page.screenshot({ path: file, fullPage: true });
    console.log(file);
  }

  async orchestrate() {
    const { chromium } = require("playwright-core");
    fs.mkdirSync(this.outDir, { recursive: true });

    const browser = await chromium.launch({ executablePath: this.resolveChromium() });
    const page    = await browser.newPage({ viewport: { width: this.width, height: this.height } });

    const base = `http://127.0.0.1:${this.port}/`;
    await page.goto(base, { waitUntil: "networkidle" });
    await page.waitForSelector("#view");

    for (const route of this.routes) await this.snapRoute(page, route);

    await browser.close();
  }
}

new UiSnapper(process.argv.slice(2)).orchestrate().catch((err) => {
  console.error(err.message || err);
  process.exit(1);
});
