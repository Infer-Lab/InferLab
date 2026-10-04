// Live updates for a workspace page (ADR-0056): each published generation
// re-renders the console region in place through htmx.
(() => {
  const region = document.getElementById("console");
  if (!region || !region.dataset.events) return;
  const stream = new EventSource(region.dataset.events);
  const rerender = () =>
    htmx.ajax("GET", window.location.href, { target: "#console", select: "#console", swap: "outerHTML" });
  stream.addEventListener("generation", rerender);
  stream.addEventListener("tick", rerender);
})();

// Browser notifications (RFC-0012:C-VIEWS): off until the operator enables
// them in this browser; then a job that ends or a running server whose
// observed process dies raises one notification.
(() => {
  const toggle = document.getElementById("notify");
  if (!toggle || !("Notification" in window)) return;
  const KEY = "inferlab-notifications";
  const enabled = () => localStorage.getItem(KEY) === "on" && Notification.permission === "granted";
  const show = () => {
    const on = enabled();
    toggle.textContent = Notification.permission === "denied" ? "Notifications blocked" : on ? "Notifications on" : "Notifications off";
    toggle.setAttribute("aria-pressed", String(on));
    toggle.hidden = false;
  };
  toggle.addEventListener("click", async () => {
    if (enabled()) {
      localStorage.removeItem(KEY);
    } else if ((await Notification.requestPermission()) === "granted") {
      localStorage.setItem(KEY, "on");
    }
    previous = null;
    show();
  });
  show();

  let previous = null;
  const notify = (title, body, href, tag) => {
    const notification = new Notification(title, { body, tag });
    notification.onclick = () => {
      window.focus();
      window.location.href = href;
    };
  };
  const poll = async () => {
    if (!enabled()) return;
    const response = await fetch("/watch", { credentials: "same-origin" }).catch(() => null);
    if (!response || !response.ok) return;
    const watch = await response.json();
    const jobs = new Map(watch.jobs.map((job) => [job.href, job]));
    const servers = new Map(watch.servers.map((server) => [server.href, server]));
    if (previous) {
      for (const [href, job] of jobs) {
        if (previous.jobs.get(href)?.running && !job.running) {
          notify(`${job.workspace}: ${job.action} ${job.state}`, job.id, href, href);
        }
      }
      for (const [href, server] of servers) {
        if (previous.servers.get(href)?.alive === true && server.alive === false) {
          notify(`${server.workspace}: server process died`, server.id, href, href);
        }
      }
    }
    previous = { jobs, servers };
  };
  poll();
  setInterval(poll, 3000);
})();
