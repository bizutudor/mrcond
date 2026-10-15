export const BANNER_TEXT =
  "Research prototype — not a medical device. Never use as the sole basis for an MRI safety decision.";

export function Banner() {
  return (
    <div className="banner" role="note" aria-label="Disclaimer">
      <svg className="banner-icon" viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
        <path d="M12 2.5 22 20.5H2Z" fill="var(--caution)" />
        <rect x="11" y="8.5" width="2" height="6.5" fill="var(--ink)" />
        <rect x="11" y="16.4" width="2" height="2" fill="var(--ink)" />
      </svg>
      <span>{BANNER_TEXT}</span>
    </div>
  );
}
