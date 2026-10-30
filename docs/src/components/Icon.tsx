/** Small, consistent line icons for workspace navigation and actions. */
export type IconName = "overview" | "papers" | "runs" | "game" | "activity" | "arrow" | "search" | "menu" | "close";
const paths: Record<IconName, string> = {
  overview: "M3 3h7v7H3z M14 3h7v7h-7z M3 14h7v7H3z M14 14h7v7h-7z",
  papers: "M5 3h10l4 4v14H5z M15 3v5h4 M8 12h8 M8 16h8",
  runs: "M8 3h12v15H8z M4 7v14h12 M11 8h6 M11 12h6",
  game: "M7 7h10c3 0 4 3 4 7s-2 5-4 2l-1-1H8l-1 1c-2 3-4 2-4-2s1-7 4-7Z M7 10v4 M5 12h4 M16 11h.01 M18 13h.01",
  activity: "M3 12h4l3-8 4 16 3-8h4",
  arrow: "M5 12h14 M14 7l5 5-5 5",
  search: "M21 21l-5-5 M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0",
  menu: "M4 6h16 M4 12h16 M4 18h16",
  close: "M6 6l12 12 M18 6 6 18",
};
export default function Icon({ name }: { name: IconName }) {
  return <svg className="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={paths[name]} /></svg>;
}
