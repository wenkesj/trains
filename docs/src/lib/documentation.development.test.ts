import {afterEach, describe, expect, it, vi} from 'vitest';
import {API_SOURCES, DEFAULT_API_URL, documentationIndex, guideOutline, loadApi, operationSlug, searchDocumentation, type ApiDocument, type LoadedApi} from './documentation';
const schema: ApiDocument = {openapi:'3.1.0',info:{title:'Trains',version:'1'},paths:{'/api/v1/create_session':{get:{summary:'List sessions'},post:{summary:'Create session'}},'/health':{get:{summary:'Health'}}}};
const api: LoadedApi = {source:API_SOURCES[0],document:schema,error:null};
afterEach(()=>{vi.useRealTimers();vi.unstubAllGlobals();vi.unstubAllEnvs();});
describe('documentation search and schema loading',()=>{
 it('searches guides and method-specific endpoint links',()=>{
  expect(searchDocumentation(documentationIndex([]),'datasets').length).toBeGreaterThan(0);
  const results=searchDocumentation(documentationIndex([api]),'POST /train/api/v1/create_session');expect(results).toHaveLength(1);
  expect(results[0].href).toBe('#/'+operationSlug('train','post','/api/v1/create_session'));
  expect(operationSlug('train','get','/api/v1/create_session')).not.toBe(operationSlug('train','post','/api/v1/create_session'));
  expect(searchDocumentation(documentationIndex([api]),'POST nonexistent')).toEqual([]);
 });
 it('ignores fenced comments in outlines',()=>{expect(guideOutline('## First\n```bash\n## Comment\n```\n## Next\n')).toEqual([{id:'first',title:'First'},{id:'next',title:'Next'}]);});
 it('loads the public schema without a key and documents the default API endpoint',async()=>{
  const fetcher=vi.fn(async(_input:RequestInfo|URL,_init?:RequestInit)=>new Response(JSON.stringify({...schema,servers:[{url:'http://internal:8380'}]})));
  vi.stubGlobal('fetch',fetcher);const loaded=await loadApi(API_SOURCES[0],new AbortController().signal);
  expect(loaded.error).toBeNull();expect(String(fetcher.mock.calls[0][0])).toBe(`${import.meta.env.BASE_URL}openapi.json`);
  expect(new Headers(fetcher.mock.calls[0][1]?.headers).has('Authorization')).toBe(false);
  expect(loaded.document?.servers).toEqual([{url:DEFAULT_API_URL}]);
expect(JSON.stringify(loaded.document)).not.toContain('test-key');
 });
 it('loads the schema beneath a GitHub Pages project path',async()=>{
  vi.stubEnv('BASE_URL','/trains/');vi.resetModules();
  const {API_SOURCES: sources,loadApi: load}=await import('./documentation');
  const fetcher=vi.fn(async()=>new Response(JSON.stringify(schema)));vi.stubGlobal('fetch',fetcher);
  const loaded=await load(sources[0],new AbortController().signal);
  expect(loaded.error).toBeNull();expect(fetcher.mock.calls[0]).toBeDefined();
  expect(sources[0].path).toBe('/trains/openapi.json');
 });
 it('rejects fallback documents while retaining searchable guides',async()=>{
  vi.stubGlobal('fetch',vi.fn(async()=>new Response(JSON.stringify({message:'app shell'}))));const result=await loadApi(API_SOURCES[0],new AbortController().signal);
  expect(result.document).toBeNull();expect(result.error).toContain('OpenAPI document');expect(documentationIndex([result]).every(r=>r.kind==='Guide')).toBe(true);
 });
 it('aborts schema loading on navigation',async()=>{
  vi.stubGlobal('fetch',vi.fn((_input:RequestInfo|URL,init?:RequestInit)=>new Promise((_resolve,reject)=>{init?.signal?.addEventListener('abort',()=>reject(init.signal?.reason),{once:true});})));
  const controller=new AbortController();const pending=loadApi(API_SOURCES[0],controller.signal);const rejected=expect(pending).rejects.toMatchObject({name:'AbortError'});controller.abort();await rejected;
 });
});
