import { useMemo } from "react";
import type { Page } from "../types";
import { segmentText, type Span } from "../lib/highlight";

interface Props {
  pages: Page[];
  spansByPage: Map<number, Span[]>;
  activeKey: string | null;
  onSelect: (key: string) => void;
  keyLabel: (key: string) => string;
}

type Item =
  | { kind: "text"; text: string }
  | { kind: "gap" }
  | { kind: "mark"; text: string; keys: string[]; anchorKeys: string[] };

// Long whitespace-only stretches come from sections masked out of multi-device or
// multilingual documents (character offsets are preserved, the text is blanked).
function isMaskedGap(text: string): boolean {
  return /^\s+$/.test(text) && (text.length > 160 || (text.match(/\n/g)?.length ?? 0) > 3);
}

export function SourceText({ pages, spansByPage, activeKey, onSelect, keyLabel }: Props) {
  const rendered = useMemo(() => {
    const anchored = new Set<string>();
    return pages.map((p) => {
      const items: Item[] = segmentText(p.text, spansByPage.get(p.number) ?? []).map((seg) => {
        if (seg.keys.length === 0) {
          return isMaskedGap(seg.text) ? { kind: "gap" } : { kind: "text", text: seg.text };
        }
        const anchorKeys = seg.keys.filter((k) => !anchored.has(k));
        anchorKeys.forEach((k) => anchored.add(k));
        return { kind: "mark", text: seg.text, keys: seg.keys, anchorKeys };
      });
      return { page: p.number, items };
    });
  }, [pages, spansByPage]);

  return (
    <div className="source">
      {rendered.map(({ page, items }) => (
        <section className="sheet" key={page} aria-label={`Page ${page}`}>
          <h3 className="sheet-page">Page {page}</h3>
          <div className="sheet-text">
            {items.map((it, i) => {
              if (it.kind === "text") return <span key={i}>{it.text}</span>;
              if (it.kind === "gap")
                return (
                  <span key={i} className="masked-gap">
                    {"\n"}text outside the MRI section is not shown{"\n"}
                  </span>
                );
              const active = activeKey !== null && it.keys.includes(activeKey);
              return (
                <mark
                  key={i}
                  className={active ? "ev ev-active" : "ev"}
                  data-anchor={it.anchorKeys.length ? it.anchorKeys.join(" ") : undefined}
                  title={it.keys.map(keyLabel).join(", ")}
                  onClick={() => onSelect(it.keys[0]!)}
                >
                  {it.text}
                </mark>
              );
            })}
          </div>
        </section>
      ))}
    </div>
  );
}
