"use strict";

class ModelGallery {
  constructor(listEl, detailEl) {
    this.listEl = listEl;
    this.detailEl = detailEl;
    this.families = [];
    this.flat = [];
    this.activeKey = null;
  }

  async load() {
    const data = await window.apiGet("/api/models");
    this.families = data.families || [];
    this.flat = this.families.flatMap((f) => f.models.map((m) => ({ ...m, family: f.family })));
    this._renderList();
    const initial = this.flat.find((m) => m.recommended) || this.flat[0];
    if (initial) this._select(initial.key);
  }

  _renderList() {
    this.listEl.innerHTML = "";
    this.families.forEach((fam) => {
      const group = document.createElement("div");
      group.className = "mlist__group";
      group.innerHTML = `<div class="mlist__label">${fam.family}</div>`;
      fam.models.forEach((m) => {
        const item = document.createElement("button");
        item.className = "mlist__item";
        item.dataset.key = m.key;
        item.innerHTML =
          `<span class="mlist__name">${m.name}${m.recommended ? '<span class="mlist__star">best</span>' : ""}</span>` +
          `<span class="mlist__params">${m.params}</span>`;
        item.addEventListener("click", () => this._select(m.key));
        group.appendChild(item);
      });
      this.listEl.appendChild(group);
    });
  }

  _select(key) {
    this.activeKey = key;
    this.listEl.querySelectorAll(".mlist__item").forEach((b) => b.classList.toggle("is-active", b.dataset.key === key));
    const model = this.flat.find((m) => m.key === key);
    if (model) this._renderDetail(model);
  }

  _renderDetail(m) {
    this._pops = this._pops || [];
    this._closeFrom(0);
    const built = window.ModelDiagram.build(m);
    this.blockMap = built.blocks;
    const legend = this._legend(m);
    this.detailEl.innerHTML =
      `<div class="mdetail__head">` +
      `<div><span class="mdetail__family">${m.family}</span>` +
      `<h3 class="mdetail__name">${m.name}${m.recommended ? '<span class="mdetail__best">best in benchmark</span>' : ""}</h3></div>` +
      `<span class="mdetail__params">${m.params} params</span>` +
      `</div>` +
      `<div class="mdetail__diagram">${built.network}</div>` +
      legend +
      `<div class="mdetail__specs">` +
      `<div class="spec"><span class="spec__k">backbone key</span><span class="spec__v mono">${m.key}</span></div>` +
      `<div class="spec"><span class="spec__k">skip mechanism</span><span class="spec__v">${m.skip}</span></div>` +
      `<div class="spec"><span class="spec__k">output head</span><span class="spec__v">${m.head}</span></div>` +
      `</div>` +
      `<div class="mdetail__when"><span class="mdetail__when-label">When to reach for it</span><p>${m.when}</p></div>`;

    this.detailEl.classList.remove("is-swap");
    void this.detailEl.offsetWidth;
    this.detailEl.classList.add("is-swap");
    this._wireSchema();
  }

  _wireSchema() {
    this._pops = this._pops || [];
    if (!this._docWired) {
      document.addEventListener("click", (e) => {
        if (this._pops.length && !e.target.closest(".dgm-pop") && !e.target.closest("[data-block]")) this._closeFrom(0);
      });
      document.addEventListener("keydown", (e) => { if (e.key === "Escape") this._closeFrom(0); });
      this._docWired = true;
    }
    this.detailEl.querySelectorAll(".mdetail__diagram [data-block]").forEach((el) => {
      el.addEventListener("click", (e) => {
        e.stopPropagation();
        const id = el.getAttribute("data-block");
        const side = id === "bridge" ? "bottom" : id === "enc" ? "left" : "right";
        this._openBlock(id, el, 0, side, null);
      });
    });
  }

  _openBlock(id, sourceEl, level, side, parentPop) {
    const def = this.blockMap[id];
    if (!def) return;
    this._closeFrom(level);

    const diagram = this.detailEl.querySelector(".mdetail__diagram");
    const pop = document.createElement("div");
    pop.className = "dgm-pop" + (def.horizontal ? " dgm-pop--wide" : "");
    pop.dataset.level = level;

    const svg = window.ModelDiagram.blockSvg(def);
    const hasSub = /data-subblock=/.test(svg);
    pop.innerHTML =
      `<div class="dgm-pop__head"><span class="dgm-pop__title">${def.title}</span>` +
      `<button class="dgm-pop__close" aria-label="Close">&times;</button></div>` +
      `<div class="dgm-pop__body">${svg}</div>` +
      (hasSub ? `<div class="dgm-pop__hint">click a highlighted block to expand it</div>` : "");

    diagram.appendChild(pop);
    sourceEl.classList.add("is-sel");
    this._positionPop(pop, parentPop || sourceEl, side, diagram, !!parentPop);
    requestAnimationFrame(() => pop.classList.add("is-in"));
    this._pops.push({ el: pop, sourceEl, level });

    pop.querySelector(".dgm-pop__close").addEventListener("click", (e) => { e.stopPropagation(); this._closeFrom(level); });
    pop.querySelectorAll("[data-subblock]").forEach((el) => {
      el.addEventListener("click", (e) => {
        e.stopPropagation();
        const sid = el.getAttribute("data-subblock");
        if (this.blockMap[sid]) this._openBlock(sid, el, level + 1, side, pop);
      });
    });
  }

  _positionPop(pop, anchorEl, side, diagram, alignTop) {
    const cR = diagram.getBoundingClientRect();
    const aR = anchorEl.getBoundingClientRect();
    const pw = pop.offsetWidth, ph = pop.offsetHeight, gap = 16;
    let vpLeft, vpTop, ox, oy;

    if (side === "bottom") {
      vpTop = aR.bottom + gap;
      vpLeft = aR.left + aR.width / 2 - pw / 2;
      ox = "50%"; oy = "0%";
    } else {
      const goLeft = side === "left";
      let l = goLeft ? aR.left - gap - pw : aR.right + gap;
      if (goLeft && l < 8) { l = aR.right + gap; ox = "0%"; }
      else if (!goLeft && l + pw > window.innerWidth - 8) { l = aR.left - gap - pw; ox = "100%"; }
      else ox = goLeft ? "100%" : "0%";
      vpLeft = l;
      vpTop = alignTop ? aR.top : aR.top + aR.height / 2 - ph / 2;
      oy = "50%";
    }

    vpLeft = Math.max(8, Math.min(vpLeft, window.innerWidth - pw - 8));
    vpTop = Math.max(74, Math.min(vpTop, window.innerHeight - ph - 8));
    pop.style.left = (vpLeft - cR.left) + "px";
    pop.style.top = (vpTop - cR.top) + "px";
    pop.style.setProperty("--ox", ox);
    pop.style.setProperty("--oy", oy);
  }

  _closeFrom(level) {
    if (!this._pops) { this._pops = []; return; }
    this._pops = this._pops.filter((p) => {
      if (p.level >= level) {
        p.el.remove();
        p.sourceEl.classList.remove("is-sel");
        return false;
      }
      return true;
    });
  }

  _legend(m) {
    const s = window.ModelDiagram.spec(m);
    const notes = {
      concat: "Skip features are concatenated into the decoder.",
      residual: "Encoder blocks carry residual connections.",
      attention: "A gate (diamond) filters skip features per region.",
      nested: "Dense intermediate nodes bridge the encoder-decoder gap.",
      additive: "Skips are summed into the decoder, not concatenated.",
      tokens: "Transformer tokens feed the CNN decoder.",
    };
    const headNote = s.heads === 1 ? "single output head emits all 3K channels" : s.heads === 3 ? "three heads split amplitude, mean, and spread" : "one head per Gaussian slot";
    return (
      `<div class="mdetail__legend">` +
      `<span class="legend__dot"></span>${notes[s.skipKind] || notes.concat}` +
      `<span class="legend__sep">/</span>${headNote}.` +
      `</div>`
    );
  }
}

window.ModelGallery = ModelGallery;
