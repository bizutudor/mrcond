import { useEffect, useState } from "react";
import { Banner } from "./components/Banner";
import { ReviewPage } from "./components/ReviewPage";
import { CheckerPage } from "./components/CheckerPage";

interface Route {
  page: "review" | "check";
  params: URLSearchParams;
}

function parseHash(hash: string): Route {
  const raw = hash.replace(/^#\/?/, "");
  const [path = "", query = ""] = raw.split("?");
  return { page: path === "check" ? "check" : "review", params: new URLSearchParams(query) };
}

export function App() {
  const [route, setRoute] = useState<Route>(() => parseHash(window.location.hash));
  useEffect(() => {
    const onHash = () => setRoute(parseHash(window.location.hash));
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  return (
    <div className="app">
      <Banner />
      <header className="topbar">
        <a className="wordmark" href="#/review" aria-label="MRCond home">
          <span className="wordmark-mark" aria-hidden="true">
            <svg viewBox="0 0 24 24" width="22" height="22">
              <path d="M5 2.5h10l4 4v15H5Z" fill="var(--paper)" stroke="var(--ink)" strokeWidth="1.6" />
              <rect x="7.5" y="9" width="9" height="3" fill="var(--caution)" />
              <path d="M7.5 15.5h9M7.5 18.5h6" stroke="var(--ink)" strokeWidth="1.3" />
            </svg>
          </span>
          MRCond
        </a>
        <nav className="tabs" aria-label="Main">
          <a href="#/review" aria-current={route.page === "review" ? "page" : undefined}>
            Review labeling
          </a>
          <a href="#/check" aria-current={route.page === "check" ? "page" : undefined}>
            Check an exam
          </a>
        </nav>
        <p className="topbar-note">Evidence-grounded MR Conditional labeling, for review by MR safety staff</p>
      </header>
      <main className="page">
        {route.page === "review" ? (
          <ReviewPage initialDocId={route.params.get("doc")} />
        ) : (
          <CheckerPage />
        )}
      </main>
    </div>
  );
}
