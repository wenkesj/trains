import {cleanup, fireEvent, render, screen, within} from '@testing-library/react';
import {afterEach,beforeEach,expect,it,vi} from 'vitest';
import Documentation from './Documentation';
vi.mock('./components/ApiDocumentation',()=>({default:({api}:{api:{source:{title:string}}})=><div>Live {api.source.title} reference</div>}));
const schema={openapi:'3.1.0',info:{title:'Trains',version:'1'},paths:{'/api/v1/create_session':{post:{summary:'Create session'}}}};
const pythonModule={id:'trains.datasets.chat',source:'trains/trains/datasets/chat.py',summary:'Reusable chat datasets.',objects:[{id:'trains.datasets.chat.chat_dataset_from_records',name:'chat_dataset_from_records',level:3}]};
const pythonContent={id:pythonModule.id,html:'<div class="doc doc-object doc-function"><h3 id="trains.datasets.chat.chat_dataset_from_records">chat_dataset_from_records</h3><p>Materialize complete conversations.</p><details><summary>Source code</summary><pre>def chat_dataset_from_records(records): ...</pre></details></div>'};
const referenceFetch=async(input:RequestInfo|URL)=>new Response(JSON.stringify(String(input).endsWith('python-reference/index.json')?{modules:[pythonModule]}:String(input).includes('python-reference/')?pythonContent:schema));
beforeEach(()=>{
 vi.stubGlobal('fetch',vi.fn(referenceFetch));vi.spyOn(window,'scrollTo').mockImplementation(()=>{});
 Object.defineProperty(HTMLDialogElement.prototype,'showModal',{configurable:true,value(){this.open=true;}});
 Object.defineProperty(HTMLDialogElement.prototype,'close',{configurable:true,value(){this.open=false;}});
});
afterEach(()=>{cleanup();vi.restoreAllMocks();vi.unstubAllGlobals();vi.unstubAllEnvs();});
it('renders the requested hierarchy without development APIs',async()=>{
 render(<Documentation param="guides/overview"/>);
 const nav=screen.getByRole('navigation',{name:'Documentation navigation'});
 for(const label of ['Overview','Benchmarks','Datum','Examples','Tracing','Reference','Python','Harbor','Harvey','Inspect','Ultrachat','Claude Agents SDK','Codex','DeepAgents','OpenAI Agents SDK','OpenCode']) expect(within(nav).getAllByText(label).length).toBeGreaterThan(0);
 expect(screen.queryByRole('button',{name:'Training'})).toBeNull();
 expect(within(nav).queryByText('Datasets')).toBeNull();
 expect(nav.querySelector('details')).toBeNull();
 expect([...nav.querySelector('.docs-nav-primary')!.querySelectorAll('a')].map(link=>link.textContent)).toEqual(['Overview','Tracing','Sessions','Benchmarks','Datum']);
 await screen.findByRole('button',{name:'Python'});
 expect(vi.mocked(fetch).mock.calls.every(([url])=>String(url).includes('python-reference/'))).toBe(true);
});
it('rejects development reference routes',()=>{
 render(<Documentation param="reference/train"/>);
 expect(screen.getByRole('heading',{name:'Documentation page not found'})).toBeTruthy();
});
it('embeds Python symbols in the shared shell',async()=>{
 render(<Documentation param="reference/python/trains.datasets.chat/trains.datasets.chat.chat_dataset_from_records"/>);
 expect(await screen.findByText('Materialize complete conversations.')).toBeTruthy();
 expect(screen.getByRole('button',{name:'Python'}).getAttribute('aria-expanded')).toBe('true');
 expect(document.querySelector('iframe')).toBeNull();
});
it('renders example tables as HTML',()=>{
 render(<Documentation param="guides/examples"/>);
 expect(screen.getByRole('table')).toBeTruthy();
 expect(screen.getAllByRole('columnheader').length).toBeGreaterThan(1);
});
it.each(['quickstart','datasets'])('merges the %s guide into Overview',id=>{
 render(<Documentation param={`guides/${id}`}/>);
 expect(screen.getByRole('heading',{name:'Overview',level:1})).toBeTruthy();
 expect(screen.getByRole('tab',{name:'pip'})).toBeTruthy();
});
it('highlights Python examples on the Datum page',()=>{
 render(<Documentation param="guides/datum"/>);
 expect(document.querySelector('pre code.language-python .hljs-keyword')).toBeTruthy();
});
it('switches installation methods with clicks and keyboard navigation',()=>{
 render(<Documentation param="guides/overview"/>);
 const pip=screen.getByRole('tab',{name:'pip'});
 const uv=screen.getByRole('tab',{name:'uv'});
 const poetry=screen.getByRole('tab',{name:'Poetry'});
 expect(pip.getAttribute('aria-selected')).toBe('true');
 expect(screen.getByRole('tabpanel',{name:'pip'}).textContent).toBe('pip install "trains-sdk @ git+https://github.com/wenkesj/trains.git#subdirectory=trains"');
 fireEvent.click(uv);
 expect(screen.getByRole('tabpanel',{name:'uv'}).textContent).toBe('uv add "trains-sdk @ git+https://github.com/wenkesj/trains.git#subdirectory=trains"');
 expect(screen.queryByRole('tabpanel',{name:'pip'})).toBeNull();
 fireEvent.keyDown(uv,{key:'ArrowRight'});
 expect(document.activeElement).toBe(poetry);
 expect(screen.getByRole('tabpanel',{name:'Poetry'}).textContent).toBe('poetry add "git+https://github.com/wenkesj/trains.git#subdirectory=trains"');
 fireEvent.keyDown(poetry,{key:'ArrowRight'});
 expect(document.activeElement).toBe(pip);
 fireEvent.keyDown(pip,{key:'End'});
 expect(document.activeElement).toBe(poetry);
 fireEvent.keyDown(poetry,{key:'Home'});
 expect(document.activeElement).toBe(pip);
 expect(document.querySelector('.docs-prose')?.textContent).not.toContain('/path/to/trains');
});
