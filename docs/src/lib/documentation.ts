import {pythonRoute, type PythonModule} from "./pythonReference";
import page0 from "../content/docs/overview.md?raw";
import page2 from "../content/docs/datasets/tracing.md?raw";
import sessions from "../content/docs/datasets/sessions.md?raw";
import page3 from "../content/docs/datasets/benchmarks.md?raw";
import page4 from "../content/docs/datasets/datum.md?raw";
import page5 from "../content/docs/examples/overview.md?raw";
import page6 from "../content/docs/examples/harbor.md?raw";
import page7 from "../content/docs/examples/harvey.md?raw";
import page8 from "../content/docs/examples/inspect.md?raw";
import page9 from "../content/docs/examples/ultrachat.md?raw";
import page10 from "../content/docs/examples/tracing/claude-agents.md?raw";
import page11 from "../content/docs/examples/tracing/codex.md?raw";
import page12 from "../content/docs/examples/tracing/deepagents.md?raw";
import page13 from "../content/docs/examples/tracing/openai-agents.md?raw";
import page14 from "../content/docs/examples/tracing/opencode.md?raw";
import hfAgentTracing from "../content/docs/examples/hf-agent-tracing.md?raw";
import trl from "../content/docs/examples/tracing/trl.md?raw";
import training from "../../development/content/training.md?raw";
import inference from "../../development/content/inference.md?raw";
export const DEFAULT_API_URL = "http://127.0.0.1:8382";
export const GUIDES = [
{id:"overview",title:"Overview",group:"",description:"",content:page0},
{id:"tracing",title:"Tracing",group:"",description:"",content:page2},
{id:"sessions",title:"Sessions",group:"",description:"",content:sessions},
{id:"benchmarks",title:"Benchmarks",group:"",description:"",content:page3},
{id:"datum",title:"Datum",group:"",description:"",content:page4},
{id:"examples",title:"Examples",group:"Examples",description:"",content:page5},
{id:"example-hf-agent-tracing",title:"HF agent tracing",group:"Examples",description:"",content:hfAgentTracing},
{id:"example-harbor",title:"Harbor",group:"Examples",description:"",content:page6},
{id:"example-harvey",title:"Harvey",group:"Examples",description:"",content:page7},
{id:"example-inspect",title:"Inspect",group:"Examples",description:"",content:page8},
{id:"example-ultrachat",title:"Ultrachat",group:"Examples",description:"",content:page9},
{id:"example-claude-agents",title:"Claude Agents SDK",group:"Tracing examples",description:"",content:page10},
{id:"example-codex",title:"Codex",group:"Tracing examples",description:"",content:page11},
{id:"example-deepagents",title:"DeepAgents",group:"Tracing examples",description:"",content:page12},
{id:"example-openai-agents",title:"OpenAI Agents SDK",group:"Tracing examples",description:"",content:page13},
{id:"example-trl",title:"Local TRL",group:"Tracing examples",description:"",content:trl},
{id:"example-opencode",title:"OpenCode",group:"Tracing examples",description:"",content:page14},
...(__DEVELOPMENT_DOCS__ ? [{id:"training",title:"Training",group:"Development",description:"",content:training},{id:"inference",title:"Inference",group:"Development",description:"",content:inference}] : []),
];
export const GUIDE_ALIASES: Record<string,string> = {datasets:"overview",quickstart:"overview",collection:"datum",sdk:"datum",integrations:"tracing",traces:"tracing"};
export interface ApiSource {id:string;title:string;path:string;prefix:string;description:string}
export const API_SOURCES: ApiSource[] = __DEVELOPMENT_DOCS__ ? [{id:"train",title:"Training",path:`${import.meta.env.BASE_URL}openapi.json`,prefix:"/train",description:"Development training API."}] : [];
export type Guide = typeof GUIDES[number];
export interface ApiOperation {
  summary?: string;
  description?: string;
  tags?: string[];
  operationId?: string;
  security?: Record<string, string[]>[];
  "x-codeSamples"?: { lang: string; label: string; source: string }[];
}
export interface ApiDocument {
  openapi: string;
  info: { title: string; version: string; description?: string };
  paths: Record<string, Record<string, ApiOperation>>;
  servers?: { url: string }[];
  components?: { securitySchemes?: Record<string, unknown>; [key: string]: unknown };
  [key: string]: unknown;
}
export interface LoadedApi {
  source: ApiSource;
  document: ApiDocument | null;
  error: string | null;
}
export interface SearchResult {
  title: string;
  detail: string;
  href: string;
  kind: string;
  text: string;
}
const METHODS = new Set(["get", "post", "put", "patch", "delete", "head", "options", "trace"]);

export function headingId(text: string): string {
  return text.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}
export function guideOutline(content: string): { id: string; title: string }[] {
  // Remove fenced code first: a commented shell line is not a document heading.
  const prose = content.replace(/^```[^\n]*\n[\s\S]*?^```\s*$/gm, "");
  return [...prose.matchAll(/^## (.+)$/gm)].map((match) => ({ id: headingId(match[1]), title: match[1] }));
}
export function operationSlug(service: string, method: string, path: string): string {
  return `docs/reference/${service}/operation/${method.toLowerCase()}-${headingId(path)}`;
}
export function apiOperations(api: LoadedApi) {
  return Object.entries(api.document?.paths ?? {}).flatMap(([path, methods]) =>
    Object.entries(methods).filter(([method]) => METHODS.has(method)).map(([method, operation]) => ({ path, method, operation })),
  );
}

/** Load the public schema with a ten-second timeout and the default self-hosted API URL. */
export async function loadApi(source: ApiSource, signal: AbortSignal): Promise<LoadedApi> {
  const controller = new AbortController();
  const abort = () => controller.abort(signal.reason);
  if (signal.aborted) abort();
  else signal.addEventListener("abort", abort, { once: true });
  const timeout = setTimeout(() => controller.abort(new Error("Schema request timed out")), 10_000);
  try {
    const response = await fetch(source.path, {signal: controller.signal});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const document = await response.json() as ApiDocument;
    if (!document || typeof document.openapi !== "string" || !document.info || !document.paths || typeof document.paths !== "object" || Array.isArray(document.paths)) {
      throw new Error("The service did not return an OpenAPI document");
    }
    return { source, document: { ...document, servers: [{ url: DEFAULT_API_URL }] }, error: null };
  } catch (error) {
    if (signal.aborted) throw error;
    return { source, document: null, error: error instanceof Error ? error.message : String(error) };
  } finally {
    clearTimeout(timeout);
    signal.removeEventListener("abort", abort);
  }
}

export function documentationIndex(apis: LoadedApi[], modules: PythonModule[] = []): SearchResult[] {
  const guides = GUIDES.map((guide) => ({
    title: guide.title, detail: guide.description, href: `#/docs/guides/${guide.id}`, kind: "Guide",
    text: `${guide.title} ${guide.description} ${guide.content}`,
  }));
  const endpoints = apis.flatMap((api) => apiOperations(api).map(({ path, method, operation }) => ({
    title: `${method.toUpperCase()} ${api.source.prefix}${path}`,
    detail: operation.summary ?? operation.description ?? api.source.description,
    href: `#/${operationSlug(api.source.id, method, path)}`, kind: `${api.source.title} API`,
    text: `${api.source.title} ${method} ${api.source.prefix}${path} ${operation.summary ?? ""} ${operation.description ?? ""} ${(operation.tags ?? []).join(" ")}`,
  })));
  const python = modules.flatMap(module => [
    {title:module.id, detail:module.summary, href:pythonRoute(module.id), kind:"Python code", text:`${module.id} ${module.summary}`},
    ...module.objects.map(object => ({title:object.id, detail:module.id, href:pythonRoute(module.id,object.id), kind:"Python code", text:`${object.id} ${object.name} ${module.id}`})),
  ]);
  return [...guides, ...endpoints, ...python];
}
export function searchDocumentation(index: SearchResult[], query: string): SearchResult[] {
  const terms = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
  if (!terms.length) return index.filter((entry) => entry.kind === "Guide");
  return index.filter((entry) => terms.every((term) => entry.text.toLowerCase().includes(term))).slice(0, 40);
}
