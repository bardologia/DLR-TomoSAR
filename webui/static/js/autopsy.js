"use strict";

class AutopsyView {
  constructor(root) {
    this.root  = root;
    this.built = false;
    this.pair  = null;
  }

  enter() {
    if (!this.built) {
      this._build();
      this.built = true;
    }
    const pending = window.abAutopsyPair;
    if (pending) {
      window.abAutopsyPair = null;
      this.refs.aInput.value = pending.a;
      this.refs.bInput.value = pending.b;
      this._compare();
    }
  }

  _build() {
    this.root.innerHTML = `
      <div class="cube-pick">
        <div class="fl-src" role="group" aria-label="Run A">
          <label for="autopsy-a">A</label>
          <input id="autopsy-a" type="text" spellcheck="false" autocomplete="off" placeholder="/path/to/run_a/inference/<stamp>" />
        </div>
        <div class="fl-src" role="group" aria-label="Run B">
          <label for="autopsy-b">B</label>
          <input id="autopsy-b" type="text" spellcheck="false" autocomplete="off" placeholder="/path/to/run_b/inference/<stamp>" />
        </div>
        <div class="fl-src">
          <button type="button" class="btn btn--mini" id="autopsy-go">Compare</button>
        </div>
        <p class="cube-hint" id="autopsy-hint">Point A and B at two saved inference stamps of the same region, or send a pair over from the leaderboard diff.</p>
      </div>
      <div id="autopsy-out"></div>`;

    this.refs = {
      aInput : this.root.querySelector("#autopsy-a"),
      bInput : this.root.querySelector("#autopsy-b"),
      goBtn  : this.root.querySelector("#autopsy-go"),
      hint   : this.root.querySelector("#autopsy-hint"),
      out    : this.root.querySelector("#autopsy-out"),
    };

    this.refs.goBtn.addEventListener("click", () => this._compare());
  }

  async _compare() {
    const a = this.refs.aInput.value.trim();
    const b = this.refs.bInput.value.trim();
    if (!a || !b) return;

    this.refs.hint.textContent = "comparing…";
    const data = await window.apiGet(`/api/autopsy/compare?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`);

    if (!data.ok) {
      this.refs.hint.textContent = data.error || "comparison failed";
      return;
    }

    this.pair = { a, b };
    this.refs.hint.textContent = `A = ${data.run_a} · B = ${data.run_b} · region ${data.region[0]}×${data.region[1]} px`;
    this._render(data);
    data.hotspots.forEach((spot, index) => this._loadProfile(spot, index));
  }

  _render(data) {
    let html = `<div class="fl-card"><h3 class="cube-panel-title">Largest metric gaps</h3>`;
    html += `<div class="lb-scroll"><table class="lb-table lb-table--diff"><thead><tr>` +
      `<th class="lb-th">metric</th><th class="lb-th">A</th><th class="lb-th">B</th><th class="lb-th">Δ rel</th><th class="lb-th">winner</th></tr></thead><tbody>`;

    data.metrics.slice(0, 25).forEach((row) => {
      const tone = row.winner === "A" ? "#0f766e" : row.winner === "B" ? "#b91c1c" : "inherit";
      html += `<tr><td class="lb-key">${this._esc(row.key)}</td>` +
        `<td>${this._fmt(row.a)}</td><td>${this._fmt(row.b)}</td>` +
        `<td>${(row.rel * 100).toFixed(1)}%</td>` +
        `<td style="color:${tone};font-weight:600">${row.winner}</td></tr>`;
    });
    html += `</tbody></table></div></div>`;

    if (data.hotspots.length) {
      html += `<div class="script-grid">`;
      data.hotspots.forEach((spot, index) => {
        html += `
          <article class="fl-card">
            <h3 class="cube-panel-title">${spot.winner} wins · az ${spot.az}, rg ${spot.rg}</h3>
            <p class="cube-hint">block Δ pixel MSE (A−B): ${spot.mean_delta.toExponential(2)}</p>
            <canvas id="autopsy-spot-${index}" width="560" height="220"></canvas>
          </article>`;
      });
      html += `</div>`;
    } else {
      html += `<p class="cube-hint">no disagreement hotspots; the runs behave almost identically pixelwise</p>`;
    }

    this.refs.out.innerHTML = html;
  }

  async _loadProfile(spot, index) {
    const data = await window.apiGet(
      `/api/autopsy/profile?a=${encodeURIComponent(this.pair.a)}&b=${encodeURIComponent(this.pair.b)}&az=${spot.az}&rg=${spot.rg}`
    );
    if (!data.ok) return;

    const canvas = this.root.querySelector(`#autopsy-spot-${index}`);
    if (!canvas) return;

    window.drawLineChart(canvas, data.x_axis, [
      { values: data.gt, color: "#111827", width: 1.2, label: "GT" },
      { values: data.a,  color: "#0f766e", width: 1.7, label: "A" },
      { values: data.b,  color: "#b91c1c", width: 1.7, label: "B" },
    ]);
  }

  _fmt(value) {
    if (!Number.isFinite(value)) return "–";
    const abs = Math.abs(value);
    if (abs !== 0 && (abs < 1e-3 || abs >= 1e4)) return value.toExponential(2);
    return value.toPrecision(4);
  }

  _esc(text) {
    return String(text).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
  }
}

window.AutopsyView = AutopsyView;
