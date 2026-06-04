"use strict";

class Router {
  constructor(onChange) {
    this.onChange = onChange;
    this.pages = {};
    document.querySelectorAll(".page").forEach((p) => {
      this.pages[p.dataset.page] = p;
    });
    this.links = [...document.querySelectorAll("[data-route]")];
    this.current = null;

    window.addEventListener("hashchange", () => this._sync());
  }

  start() {
    this._sync();
  }

  go(route) {
    window.location.hash = `#/${route}`;
  }

  _parse() {
    const raw = (window.location.hash || "").replace(/^#\/?/, "").trim();
    return this.pages[raw] ? raw : "home";
  }

  _sync() {
    const route = this._parse();
    if (route === this.current) return;
    this.current = route;

    Object.entries(this.pages).forEach(([id, el]) => {
      const active = id === route;
      el.classList.toggle("is-active", active);
    });

    this.links.forEach((a) => a.classList.toggle("is-current", a.dataset.route === route));

    window.scrollTo({ top: 0, behavior: "instant" in window ? "instant" : "auto" });
    this._animateIn(this.pages[route]);

    if (this.onChange) this.onChange(route);
  }

  _animateIn(page) {
    if (!page || window.REDUCED_MOTION || !window.gsap) return;
    const blocks = page.querySelectorAll(".page__head, .page__head > *, .eq-tabs, .eq-grid, .flow, .master, .filter-row, .script-grid, .console, .hero__inner");
    const targets = blocks.length ? blocks : [page];
    window.gsap.fromTo(
      targets,
      { opacity: 0, y: 18 },
      { opacity: 1, y: 0, duration: 0.5, ease: "power3.out", stagger: 0.05, overwrite: true }
    );
  }
}

window.Router = Router;
