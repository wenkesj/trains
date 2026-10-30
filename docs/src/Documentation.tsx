/** Searchable guides and live Scalar reference in a shared developer documentation shell. */
import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import Icon from "./components/Icon";
import InstallationTabs from "./components/InstallationTabs";
import PythonDocumentation from "./components/PythonDocumentation";
import {loadPythonModules, pythonGroups, pythonRoute, type PythonModule} from "./lib/pythonReference";
import {
  API_SOURCES, GUIDES, GUIDE_ALIASES, apiOperations, documentationIndex, guideOutline, headingId,
  loadApi, operationSlug, searchDocumentation, type LoadedApi,
} from "./lib/documentation";
import "./documentation.css";

const ApiDocumentation = __DEVELOPMENT_DOCS__ ? lazy(() => import("./components/ApiDocumentation")) : null;

export default function Documentation({ param }: { param: string | null }) {
  const [mode = "guides", rawRequested = "overview", selectedModule, selectedObject] = (param ?? "guides/overview").split("/");
  const requested = GUIDE_ALIASES[rawRequested] ?? rawRequested;
  const reference = mode === "reference";
  const python = reference && requested === "python";
  const referencePath = param?.split("/").slice(0, 4).join("/");
  const guide = GUIDES.find((item) => item.id === requested);
  const source = API_SOURCES.find((item) => item.id === requested);
  const valid = reference ? python || !!source : mode === "guides" && !!guide;
  const [apis, setApis] = useState<LoadedApi[]>([]);
  const [reload, setReload] = useState(0);
  const [modules, setModules] = useState<PythonModule[] | null>(null);
  const [pythonError, setPythonError] = useState<string | null>(null);
  const [expandedReference, setExpandedReference] = useState<string | null>(reference ? requested : null);
  const [navOpen, setNavOpen] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const [query, setQuery] = useState("");
  const searchDialog = useRef<HTMLDialogElement>(null);
  const content = useRef<HTMLElement>(null);
  const api = apis.find((item) => item.source.id === requested);
  const module = modules?.find(item => item.id === selectedModule);
  const groups = useMemo(() => pythonGroups(modules ?? []), [modules]);
  const index = useMemo(() => documentationIndex(apis, modules ?? []), [apis, modules]);
  const results = useMemo(() => searchDocumentation(index, query), [index, query]);
  const outline = guide ? guideOutline(guide.content) : [];
  const guideAt = GUIDES.findIndex((item) => item.id === requested);

  useEffect(() => {
    const controller = new AbortController();
    setApis([]);setModules(null);setPythonError(null);
    loadPythonModules(controller.signal).then(value=>{
      if (!controller.signal.aborted) setModules(value);
    }).catch(error=>{if (!controller.signal.aborted) setPythonError(error instanceof Error ? error.message : String(error));});
    if (__DEVELOPMENT_DOCS__) API_SOURCES.forEach((item) => {
      loadApi(item, controller.signal).then((loaded) => {
        if (!controller.signal.aborted) setApis((current) => [...current, loaded]);
      }).catch(() => { /* Aborted navigation has no result to display. */ });
    });
    return () => controller.abort();
  }, [reload]);

  useEffect(() => {
    setExpandedReference(reference ? requested : null);
  }, [mode, requested]);

  useEffect(() => {
    setNavOpen(false);
    window.scrollTo?.(0, 0);
  }, [mode, requested, param]);

  useEffect(() => {
    if (searchOpen) searchDialog.current?.showModal();
    else if (searchDialog.current?.open) searchDialog.current.close();
  }, [searchOpen]);

  useEffect(() => {
    const shortcut = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setSearchOpen((open) => !open);
      }
      if (event.key === "Escape") setNavOpen(false);
    };
    window.addEventListener("keydown", shortcut);
    return () => window.removeEventListener("keydown", shortcut);
  }, []);

  const jump = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth", block: "start" });
  };
  const closeSearch = () => { setSearchOpen(false); setQuery(""); };
  const toggleReference = (name: string) => {
    const expanded = expandedReference !== name;
    setExpandedReference(expanded ? name : null);
    if (expanded && (!reference || requested !== name)) window.location.hash = name === "python" ? pythonRoute() : `#/docs/reference/${name}`;
  };

  return <div className="docs-shell">
    <a className="skip-link" href="#docs-content" onClick={(event) => { event.preventDefault(); content.current?.focus(); }}>Skip to content</a>
    <header className="docs-topbar">
      <button className="docs-menu" aria-label="Toggle documentation navigation" aria-expanded={navOpen} aria-controls="docs-navigation" onClick={() => setNavOpen(!navOpen)}><Icon name="menu" /></button>
      <span className="docs-product">Developer docs</span>
      <button className="docs-search-trigger" aria-label="Search docs" onClick={() => setSearchOpen(true)}><Icon name="search" /><span>Search docs</span><kbd>⌘ / Ctrl K</kbd></button>
      <a className="docs-github" href="https://github.com/wenkesj/trains" target="_blank" rel="noreferrer" aria-label="GitHub: wenkesj/trains">
        <svg viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38v-1.49c-2.23.48-2.7-.94-2.7-.94-.36-.92-.89-1.17-.89-1.17-.73-.5.05-.49.05-.49.8.06 1.22.82 1.22.82.71 1.21 1.87.86 2.33.66.07-.51.28-.86.51-1.06-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.65 7.65 0 014 0c1.53-1.03 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.28.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48v2.2c0 .21.14.46.55.38A8.01 8.01 0 0016 8c0-4.42-3.58-8-8-8Z"/></svg>
        <span><strong>wenkesj/trains</strong><small>GitHub</small></span>
      </a>
    </header>
    {navOpen && <button className="docs-backdrop" aria-label="Close documentation navigation" onClick={() => setNavOpen(false)} />}
    <aside id="docs-navigation" className={`docs-sidebar ${navOpen ? "open" : ""}`}>
      <a className="brand" href="#/docs/guides/overview"><img src={`${import.meta.env.BASE_URL}trains.svg`} width="36" height="36" alt="" /><strong>trains</strong></a>
      <nav aria-label="Documentation navigation">
        <div className="docs-nav-group docs-nav-primary">
          {GUIDES.filter(item=>item.group==="").map(item=><a key={item.id} href={`#/docs/guides/${item.id}`} aria-current={guide?.id===item.id ? "page" : undefined}>{item.title}</a>)}
        </div>
        <div className="docs-nav-group"><h2>Examples</h2>
          {GUIDES.filter(item=>item.group==="Examples").map(item=><a key={item.id} href={`#/docs/guides/${item.id}`} aria-current={guide?.id===item.id ? "page" : undefined}>{item.id==="examples" ? "Overview" : item.title}</a>)}
          <h3 className="docs-nav-subheading">Tracing</h3>
          <div className="docs-nested">{GUIDES.filter(item=>item.group==="Tracing examples").map(item=><a key={item.id} href={`#/docs/guides/${item.id}`} aria-current={guide?.id===item.id ? "page" : undefined}>{item.title}</a>)}</div>
        </div>
        {__DEVELOPMENT_DOCS__ && <div className="docs-nav-group"><h2>Development only</h2>{GUIDES.filter(item=>item.group==="Development").map(item=><a key={item.id} href={`#/docs/guides/${item.id}`}>{item.title}</a>)}</div>}
        <div className="docs-nav-group"><h2>Reference</h2>
          <button className="docs-reference-toggle" aria-expanded={expandedReference === "python"} aria-controls="python-navigation" aria-current={python ? "page" : undefined} onClick={()=>toggleReference("python")}>
            <span>Python</span><span className="docs-chevron" aria-hidden="true">›</span>
          </button>
          {expandedReference === "python" && <div id="python-navigation" className="docs-nested" role="group" aria-label="Python code sections">
            <a href={pythonRoute()} aria-current={python && !selectedModule ? "page" : undefined}>Overview</a>
            {!modules && !pythonError && <p className="docs-nav-status">Loading modules…</p>}
            {pythonError && <button className="docs-nav-retry" onClick={()=>setReload(value=>value+1)}>Retry Python reference</button>}
            {groups.map(group=><details className="docs-module-group" key={group.name} open={!!selectedModule && group.modules.some(item=>item.id===selectedModule)}>
              <summary>{group.name === "trains" ? "Package" : group.name}</summary>
              {group.modules.map(item=><div key={item.id}>
                <a href={pythonRoute(item.id)} title={item.id} aria-current={python && selectedModule===item.id && !selectedObject ? "page" : undefined}>{item.id.replace(/^trains\./, "")}</a>
                {selectedModule === item.id && <div className="docs-symbols">{item.objects.map(object=><a key={object.id} href={pythonRoute(item.id,object.id)} title={object.id} aria-current={selectedObject===object.id ? "page" : undefined}>{object.name}</a>)}</div>}
              </div>)}
            </details>)}
          </div>}
          {API_SOURCES.map(item=><div key={item.id}>
            <a href={`#/docs/reference/${item.id}`} aria-current={reference && source?.id===item.id && !param?.includes("/operation/") ? "page" : undefined}>{item.title}</a>
            <div id={`${item.id}-navigation`} className="docs-nested docs-endpoints" role="group" aria-label={`${item.title} API sections`}>
              {apis.filter(loaded=>loaded.source.id===item.id).flatMap(loaded=>{
                const operations=apiOperations(loaded);
                return [...new Set(operations.map(entry=>entry.operation.tags?.[0] ?? "General"))].map(tag=>{
                  const entries=operations.filter(entry=>(entry.operation.tags?.[0] ?? "General")===tag);
                  return <div className="docs-module-group docs-api-group" key={tag}>
                    <h3 className="docs-nav-subheading">{tag}</h3>
                    {entries.map(({method,path,operation})=><a key={`${method}-${path}`} href={`#/${operationSlug(item.id,method,path)}`} aria-current={param===operationSlug(item.id,method,path).replace(/^docs\//, "") ? "page" : undefined} title={`${method.toUpperCase()} ${path}`}><span className={`docs-method ${method}`}>{method.toUpperCase()}</span><span>{operation.summary ?? path}</span></a>)}
                  </div>;
                });
              })}
            </div>
          </div>)}
        </div>
      </nav>
    </aside>
    <main id="docs-content" ref={content} tabIndex={-1} className={`docs-main ${reference ? "api-mode" : ""}`}>
      {!valid ? <article className="docs-article"><h1>Documentation page not found</h1><p>Choose a page from the navigation.</p><a href="#/docs/guides/overview">Go to the overview →</a></article> : python ? <>
        <div className="docs-reference-head"><p className="docs-eyebrow">Python code reference</p><h1>{selectedModule ?? "Python code"}</h1><p>Signatures, types, examples, and source from Trains's public API.</p></div>
        {pythonError ? <div className="docs-error" role="alert"><h2>Python reference unavailable</h2><p>{pythonError}</p><button onClick={()=>setReload(value=>value+1)}>Retry Python reference</button></div>
          : !modules ? <p className="docs-loading" role="status">Loading Python reference…</p>
          : selectedModule ? module ? <PythonDocumentation module={module} object={selectedObject} retry={reload}/> : <div className="docs-error"><h2>Python module not found</h2><a href={pythonRoute()}>Browse Python modules</a></div>
          : <div className="docs-python-overview">{groups.map(group=><section key={group.name}><h2>{group.name === "trains" ? "Package" : group.name}</h2><div className="docs-module-cards">{group.modules.map(item=><a href={pythonRoute(item.id)} key={item.id}><code>{item.id}</code>{item.summary && <p>{item.summary}</p>}<span>View reference →</span></a>)}</div></section>)}</div>}
      </> : __DEVELOPMENT_DOCS__ && reference && ApiDocumentation ? <>
        <div className="docs-reference-head"><p className="docs-eyebrow">API reference</p><h1>{source?.title} API</h1><p>{source?.description}</p></div>
        {!api ? <p className="docs-loading" role="status">Loading the service's OpenAPI document…</p> : api.error ? <div className="docs-error" role="alert"><h2>{source?.title} reference unavailable</h2><p>Could not load the service schema: {api.error}. Check that the service is running, then retry.</p><button onClick={() => setReload((value) => value + 1)}>Retry schema</button><a href="#/docs/guides/overview">Read the guides →</a></div> : <Suspense fallback={<p className="docs-loading" role="status">Loading API reference…</p>}><ApiDocumentation key={referencePath ?? source?.id} api={api} focus={referencePath ? `docs/${referencePath}` : undefined} /></Suspense>}
      </> : <div className="docs-reading-layout">
        <article className="docs-article">
          <p className="docs-eyebrow">{guide?.group}</p><h1>{guide?.title}</h1><p className="docs-lead">{guide?.description}</p>
          {guide?.id === "overview" && <div className="docs-badges" aria-label="Project status">
            <a href="https://github.com/wenkesj/trains/actions/workflows/ci.yml"><img src="https://github.com/wenkesj/trains/actions/workflows/ci.yml/badge.svg" alt="CI status" /></a>
            <a href="https://github.com/wenkesj/trains/actions/workflows/pages.yml"><img src="https://github.com/wenkesj/trains/actions/workflows/pages.yml/badge.svg" alt="Documentation build status" /></a>
            <a href="#/docs/guides/quickstart"><img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python 3.12 or later" /></a>
          </div>}
          <div className="docs-prose"><Markdown remarkPlugins={[remarkGfm]} rehypePlugins={[[rehypeHighlight, {ignoreMissing: true}]]} components={{
            h2: ({ children }) => <h2 id={headingId(String(children))}>{children}</h2>,
            table: ({ children }) => <div className="docs-table"><table>{children}</table></div>,
            pre: ({ children, node }) => {
              const code = node?.children.find(child=>child.type==="element" && child.tagName==="code");
              if (code?.type==="element" && Array.isArray(code.properties.className) && code.properties.className.includes("language-installation")) {
                const commands = code.children.map(child=>child.type==="text" ? child.value : "").join("").trim().split("\n");
                return <InstallationTabs commands={commands}/>;
              }
              return <div className="docs-code"><pre>{children}</pre></div>;
            },
          }}>{guide?.content ?? ""}</Markdown></div>
          <nav className="docs-pagination" aria-label="Adjacent guides">
            {GUIDES[guideAt - 1] ? <a href={`#/docs/guides/${GUIDES[guideAt - 1].id}`}><small>Previous</small>{GUIDES[guideAt - 1].title}</a> : <span />}
            {GUIDES[guideAt + 1] && <a href={`#/docs/guides/${GUIDES[guideAt + 1].id}`}><small>Next</small>{GUIDES[guideAt + 1].title} →</a>}
          </nav>
        </article>
        <aside className="docs-outline" aria-label="On this page"><p>On this page</p>{outline.map((heading) => <button key={heading.id} onClick={() => jump(heading.id)}>{heading.title}</button>)}</aside>
      </div>}
    </main>
    <dialog className="docs-search-dialog" aria-label="Search documentation" ref={searchDialog} onClose={closeSearch} onCancel={closeSearch}>
      <div className="docs-search-input"><Icon name="search" /><input aria-label="Search documentation" placeholder="Search guides and Python reference…" value={query} onChange={(event) => setQuery(event.target.value)} autoFocus /><button aria-label="Close search" onClick={closeSearch}><kbd>Esc</kbd></button></div>
      <div className="docs-search-results" aria-live="polite">
        {!query && <p className="docs-search-caption">Suggested guides</p>}
        {results.length ? results.map((result) => <a key={result.href} href={result.href} onClick={closeSearch}><small>{result.kind}</small><strong>{result.title}</strong><span>{result.detail}</span><Icon name="arrow" /></a>) : <p className="docs-search-empty">No results for “{query}”. Try a module name or concept.</p>}
      </div>
      <div className="docs-search-footer">Search guides and Python reference.{apis.some((item) => item.error) && <span> Some service schemas are unavailable.</span>}</div>
    </dialog>
  </div>;
}
