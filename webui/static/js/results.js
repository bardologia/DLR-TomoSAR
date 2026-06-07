"use strict";

class ResultsView {
  constructor(listEl, detailEl) {
    this.listEl    = listEl;
    this.detailEl  = detailEl;
    this.pathForm  = document.getElementById("results-path-form");
    this.pathInput = document.getElementById("results-path-input");

    this.root      = null;
    this.activeRel = null;
    this.expanded  = new Set();
    this.loaded    = false;

    this.lightbox    = document.getElementById("lightbox");
    this.lightboxImg = document.getElementById("lightbox-img");
    this.lightboxCap = document.getElementById("lightbox-cap");
    this.lightbox.addEventListener("click", () => this._closeLightbox());
    document.addEventListener("keydown", (ev) => {
      if (ev.key === "Escape") this._closeLightbox();
    });

    this.pathForm.addEventListener("submit", (ev) => {
      ev.preventDefault();
      const path = this.pathInput.value.trim();
      if (path) this.open(path);
    });
  }

  enter() {
    if (this.loaded) return;
    this.loaded = true;

    const recents = this._recents();
    if (recents.length) {
      this.pathInput.value = recents[0];
      this.open(recents[0]);
    } else {
      this._renderSidebar();
      this.detailEl.innerHTML = `<div class="res-empty">Paste the path of any run directory above &mdash; preprocessing, parameter extraction, training, inference or benchmark &mdash; and press Open.</div>`;
    }
  }

  async open(path) {
    this.detailEl.innerHTML = `<div class="res-empty">Loading&hellip;</div>`;

    let data;
    try {
      data = await window.apiGet(`/api/results/tree?path=${encodeURIComponent(path)}`);
    } catch (e) {
      data = null;
    }

    if (!data || !data.ok) {
      this.root = null;
      this._renderSidebar();
      this.detailEl.innerHTML = `<div class="res-empty">${this._esc(data && data.error ? data.error : "Could not open this path.")}</div>`;
      return;
    }

    this.root            = data;
    this.pathInput.value = data.root;
    this._remember(data.root);

    this.expanded = new Set([""]);
    data.tree.children.forEach((child) => this.expanded.add(child.rel));

    this._renderSidebar();
    this.selectFolder("");
  }

  async selectFolder(rel) {
    this.activeRel = rel;
    this._renderSidebar();
    this.detailEl.innerHTML = `<div class="res-empty">Loading&hellip;</div>`;

    let data;
    try {
      data = await window.apiGet(`/api/results/folder?root=${encodeURIComponent(this.root.root)}&rel=${encodeURIComponent(rel)}`);
    } catch (e) {
      data = null;
    }

    if (!data || !data.ok) {
      this.detailEl.innerHTML = `<div class="res-empty">Could not load this folder.</div>`;
      return;
    }

    this._renderFolder(data);
  }

  _renderSidebar() {
    this.listEl.innerHTML = "";

    const recents = this._recents();
    if (recents.length) {
      const head = document.createElement("div");
      head.className = "run-group__head";
      head.innerHTML = `<span>Recent</span><span class="run-group__count">${recents.length}</span>`;
      this.listEl.appendChild(head);

      recents.forEach((path) => {
        const item = document.createElement("button");
        item.type = "button";
        item.className = "run-item" + (this.root && this.root.root === path ? " is-active" : "");
        item.innerHTML = `<span class="run-item__name">${this._esc(this._shorten(path))}</span>`;
        item.title = path;
        item.addEventListener("click", () => this.open(path));
        this.listEl.appendChild(item);
      });
    }

    if (!this.root) {
      if (!recents.length) this.listEl.innerHTML = `<div class="clist__empty">No results opened yet.</div>`;
      return;
    }

    const head = document.createElement("div");
    head.className = "run-group__head";
    head.innerHTML = `<span>${this._esc(this.root.name)}</span><span class="res-stage">${this._esc(this.root.stage)}</span>`;
    this.listEl.appendChild(head);

    this._renderNode(this.root.tree, 0);
  }

  _renderNode(node, depth) {
    const total  = node.counts.markdown + node.counts.images + node.counts.animations + node.counts.configs;
    const isOpen = this.expanded.has(node.rel);

    const row = document.createElement("div");
    row.className = "res-tree__row" + (node.rel === this.activeRel ? " is-active" : "");
    row.style.paddingLeft = `${8 + depth * 14}px`;

    const caret = document.createElement("span");
    caret.className = "res-tree__caret" + (node.children.length ? (isOpen ? " is-open" : "") : " is-leaf");
    caret.textContent = node.children.length ? "▸" : "";
    caret.addEventListener("click", (ev) => {
      ev.stopPropagation();
      if (isOpen) this.expanded.delete(node.rel);
      else this.expanded.add(node.rel);
      this._renderSidebar();
    });

    const name = document.createElement("span");
    name.className = "res-tree__name";
    name.textContent = node.rel === "" ? "/" : node.name;

    row.append(caret, name);

    if (total) {
      const badge = document.createElement("span");
      badge.className = "res-tree__count";
      badge.textContent = total;
      row.appendChild(badge);
    }

    row.addEventListener("click", () => this.selectFolder(node.rel));
    this.listEl.appendChild(row);

    if (isOpen) node.children.forEach((child) => this._renderNode(child, depth + 1));
  }

  _renderFolder(data) {
    let html =
      `<header class="res-head">` +
      `<div><h3 class="res-title">${this._esc(data.rel === "" ? this.root.name : data.rel)}</h3>` +
      `<p class="res-path">${this._esc(data.abs)}</p></div>` +
      `<span class="res-stamp is-active is-single">${this._esc(this.root.stage)}</span>` +
      `</header>`;

    data.markdown.forEach((md, index) => {
      html += `<section class="res-section">`;
      html += `<h4 class="res-section__cap">${this._esc(md.name)}</h4>`;
      html += `<article class="res-md" data-md="${index}"></article>`;
      html += `</section>`;
    });

    if (data.images.length)     html += this._sectionHtml("Plots", data.images, "img");
    if (data.animations.length) html += this._sectionHtml("Animations", data.animations, "gif");

    if (data.configs.length) {
      html += `<section class="res-section"><h4 class="res-section__cap">Configs <span>${data.configs.length}</span></h4>`;
      data.configs.forEach((cfg, index) => {
        html += `<details class="res-cfg"${data.configs.length === 1 ? " open" : ""}>`;
        html += `<summary>${this._esc(cfg.name)}</summary>`;
        html += `<pre><code data-cfg="${index}"></code></pre>`;
        html += `</details>`;
      });
      html += `</section>`;
    }

    if (data.other.length) {
      html += `<section class="res-section"><h4 class="res-section__cap">Other files <span>${data.other.length}</span></h4><ul class="res-files">`;
      data.other.forEach((file) => {
        html += `<li><span>${this._esc(file.name)}</span><span>${this._size(file.size)}</span></li>`;
      });
      html += `</ul></section>`;
    }

    if (!data.markdown.length && !data.images.length && !data.animations.length && !data.configs.length && !data.other.length) {
      html += `<div class="res-empty">This folder holds no files. Pick a subfolder on the left.</div>`;
    }

    this.detailEl.innerHTML = html;
    this._fillMarkdown(data);
    this._fillConfigs(data);
    this._bindFigures();
  }

  _fillMarkdown(data) {
    this.detailEl.querySelectorAll(".res-md").forEach((body) => {
      const md = data.markdown[Number(body.dataset.md)];

      let parsed;
      try {
        parsed = window.marked ? window.marked.parse(md.text) : null;
      } catch (e) {
        parsed = null;
      }

      if (parsed === null) {
        body.innerHTML = `<pre>${this._esc(md.text)}</pre>`;
        return;
      }

      body.innerHTML = parsed;
      this._gridify(body);

      body.querySelectorAll("img").forEach((img) => {
        const src = img.getAttribute("src") || "";
        if (!src.startsWith("/") && !src.startsWith("http") && !src.startsWith("data:")) {
          img.src = this._mediaUrl(data.abs, src);
        }
        img.loading = "lazy";
      });

      body.querySelectorAll("a").forEach((a) => {
        const href = a.getAttribute("href") || "";
        if (!href.startsWith("/") && !href.startsWith("http") && !href.startsWith("#")) {
          a.href = this._mediaUrl(data.abs, href);
        }
        a.target = "_blank";
        a.rel = "noopener";
      });
    });
  }

  _fillConfigs(data) {
    this.detailEl.querySelectorAll("code[data-cfg]").forEach((code) => {
      const cfg = data.configs[Number(code.dataset.cfg)];
      code.textContent = cfg.kind === "json" ? this._prettyJson(cfg.text) : cfg.text;
    });
  }

  _bindFigures() {
    this.detailEl.querySelectorAll(".res-fig[data-url]").forEach((fig) => {
      fig.addEventListener("click", () => this._openLightbox(fig.dataset.url, fig.dataset.name));
    });
    this.detailEl.querySelectorAll(".res-md img").forEach((img) => {
      img.addEventListener("click", () => this._openLightbox(img.src, img.alt || ""));
    });
  }

  _sectionHtml(title, items, kind) {
    const figs = items.map((item) =>
      `<figure class="res-fig${kind === "gif" ? " res-fig--gif" : ""}" data-url="${item.url}" data-name="${this._esc(item.name)}">` +
      `<img src="${item.url}" alt="${this._esc(item.name)}" loading="lazy" />` +
      `<figcaption>${this._esc(item.name)}</figcaption>` +
      `</figure>`
    ).join("");

    return (
      `<section class="res-section">` +
      `<h4 class="res-section__cap">${this._esc(title)} <span>${items.length}</span></h4>` +
      `<div class="res-grid${kind === "gif" ? " res-grid--gif" : ""}">${figs}</div>` +
      `</section>`
    );
  }

  _gridify(body) {
    const isImgPara = (el) =>
      el && el.tagName === "P" && el.children.length === 1 && el.children[0].tagName === "IMG" && !el.textContent.trim();

    let node = body.firstElementChild;
    while (node) {
      if (!isImgPara(node)) {
        node = node.nextElementSibling;
        continue;
      }

      const run = [node];
      let next = node.nextElementSibling;
      while (isImgPara(next)) {
        run.push(next);
        next = next.nextElementSibling;
      }

      const grid = document.createElement("div");
      grid.className = "res-grid res-grid--md";
      node.before(grid);

      run.forEach((para) => {
        const img = para.children[0];
        const isGif = (img.getAttribute("src") || "").toLowerCase().endsWith(".gif");

        const fig = document.createElement("figure");
        fig.className = "res-fig" + (isGif ? " res-fig--gif" : "");

        const cap = document.createElement("figcaption");
        cap.textContent = img.alt || "";

        fig.append(img, cap);
        grid.appendChild(fig);
        para.remove();
      });

      node = grid.nextElementSibling;
    }
  }

  _mediaUrl(folderAbs, relative) {
    return "/resultsmedia?path=" + encodeURIComponent(folderAbs + "/" + decodeURIComponent(relative));
  }

  _prettyJson(text) {
    try {
      return JSON.stringify(JSON.parse(text), null, 2);
    } catch (e) {
      return text;
    }
  }

  _recents() {
    try {
      return JSON.parse(localStorage.getItem("results-recents") || "[]");
    } catch (e) {
      return [];
    }
  }

  _remember(path) {
    const recents = [path, ...this._recents().filter((p) => p !== path)].slice(0, 8);
    localStorage.setItem("results-recents", JSON.stringify(recents));
  }

  _shorten(path) {
    const parts = path.replace(/\/+$/, "").split("/");
    return parts.length > 3 ? "…/" + parts.slice(-3).join("/") : path;
  }

  _size(bytes) {
    if (bytes >= 1073741824) return (bytes / 1073741824).toFixed(1) + " GB";
    if (bytes >= 1048576)    return (bytes / 1048576).toFixed(1) + " MB";
    if (bytes >= 1024)       return (bytes / 1024).toFixed(1) + " KB";
    return bytes + " B";
  }

  _openLightbox(url, name) {
    this.lightboxImg.src = url;
    this.lightboxCap.textContent = name;
    this.lightbox.hidden = false;
  }

  _closeLightbox() {
    if (this.lightbox.hidden) return;
    this.lightbox.hidden = true;
    this.lightboxImg.src = "";
  }

  _esc(text) {
    return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
}

window.ResultsView = ResultsView;
