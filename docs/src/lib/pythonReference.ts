/** Navigation and lazy content from statically generated Python docstrings. */
export interface PythonObject { id: string; name: string; level: number }
export interface PythonModule { id: string; summary: string; source: string; objects: PythonObject[] }
export interface PythonContent { id: string; html: string }

export function pythonRoute(module?: string, object?: string): string {
  return '#/docs/reference/python' + (module ? `/${encodeURIComponent(module)}` : '') + (object ? `/${encodeURIComponent(object)}` : '');
}

async function readReference(file: string, signal: AbortSignal): Promise<unknown> {
  const controller = new AbortController();
  const abort = () => controller.abort(signal.reason);
  if (signal.aborted) abort();
  else signal.addEventListener('abort', abort, {once: true});
  const timeout = setTimeout(() => controller.abort(new Error('Python reference request timed out')), 10_000);
  try {
    const response = await fetch(`${import.meta.env.BASE_URL}python-reference/${file}`, {signal: controller.signal});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } finally {
    clearTimeout(timeout);
    signal.removeEventListener('abort', abort);
  }
}

export async function loadPythonModules(signal: AbortSignal): Promise<PythonModule[]> {
  const value = await readReference('index.json', signal) as {modules?: PythonModule[]};
  if (!Array.isArray(value.modules) || !value.modules.length || value.modules.some(module => typeof module.id !== 'string' || typeof module.summary !== 'string' || typeof module.source !== 'string' || !Array.isArray(module.objects) || module.objects.some(object => typeof object.id !== 'string' || typeof object.name !== 'string' || typeof object.level !== 'number'))) {
    throw new Error('The Python reference manifest is invalid');
  }
  return value.modules;
}

export async function loadPythonContent(module: string, signal: AbortSignal): Promise<PythonContent> {
  const value = await readReference(`${encodeURIComponent(module)}.json`, signal) as PythonContent;
  if (value.id !== module || typeof value.html !== 'string') throw new Error('The Python reference module is invalid');
  return value;
}

export function pythonGroups(modules: PythonModule[]): {name: string; modules: PythonModule[]}[] {
  const groups = new Map<string, PythonModule[]>();
  for (const module of modules) {
    const name = module.id.split('.')[1] ?? 'trains';
    groups.set(name, [...(groups.get(name) ?? []), module]);
  }
  return [...groups].map(([name, modules]) => ({name, modules}));
}
