/** Render generated signatures and docstrings using the shared API reference styling. */
import {useEffect, useState} from 'react';
import {loadPythonContent, pythonRoute, type PythonContent, type PythonModule} from '../lib/pythonReference';

export default function PythonDocumentation({module, object, retry}: {module: PythonModule; object?: string; retry: number}) {
  const [content, setContent] = useState<PythonContent | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reload, setReload] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    setContent(null);setError(null);
    loadPythonContent(module.id, controller.signal).then(value => {
      if (!controller.signal.aborted) setContent(value);
    }).catch(error => {if (!controller.signal.aborted) setError(error instanceof Error ? error.message : String(error));});
    return () => controller.abort();
  }, [module.id, retry, reload]);
  useEffect(() => {
    if (content?.id === module.id && object) document.getElementById(object)?.scrollIntoView?.({block: 'start'});
  }, [content, module.id, object]);
  if (error) return <div className="docs-error" role="alert"><h2>Python reference unavailable</h2><p>{error}</p><button onClick={()=>setReload(value=>value+1)}>Retry Python module</button></div>;
  if (content?.id !== module.id) return <p className="docs-loading" role="status">Loading Python module…</p>;
  return <>
    {object && !module.objects.some(item=>item.id===object) && <p className="docs-loading">This symbol is unavailable. Showing its module.</p>}
    {/* HTML is produced at build time from repository-owned mkdocstrings, not user input. */}
    <article className="python-reference docs-prose" aria-label={`${module.id} code reference`} dangerouslySetInnerHTML={{__html:content.html}} />
    <nav className="docs-pagination" aria-label="Python reference navigation"><a href={pythonRoute()}>← All Python modules</a><a href={`https://github.com/wenkesj/trains/blob/main/${module.source}`} target="_blank" rel="noreferrer">View repository →</a></nav>
  </>;
}
