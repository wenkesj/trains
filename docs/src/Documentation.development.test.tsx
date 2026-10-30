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
it('renders guides, page outline, and guide navigation',()=>{
 render(<Documentation param="guides/quickstart"/>);expect(screen.getByRole('heading',{name:'Overview',level:1})).toBeTruthy();
 expect(document.querySelector('.docs-outline button')).toBeTruthy();expect(screen.getByLabelText('On this page')).toBeTruthy();expect(screen.queryByText('robo')).toBeNull();
});
it('searches live endpoints with Ctrl K and preserves deep links',async()=>{
 render(<Documentation param="guides/overview"/>);fireEvent.keyDown(window,{key:'k',ctrlKey:true});fireEvent.change(screen.getByRole('textbox',{name:'Search documentation'}),{target:{value:'POST /train/api/v1/create_session'}});
 const result=await screen.findByRole('link',{name:/Training API POST \/train\/api\/v1\/create_session/});expect(result.getAttribute('href')).toBe('#/docs/reference/train/operation/post-api-v1-create-session');fireEvent.click(screen.getByRole('button',{name:'Close search'}));expect(screen.queryByRole('textbox',{name:'Search documentation'})).toBeNull();
});
it('opens an operation deep link',async()=>{render(<Documentation param="reference/train/operation/post-api-v1-create-session"/>);expect(await screen.findByText('Live Training reference')).toBeTruthy();});
it('retains guides and retries an unavailable schema',async()=>{
 vi.stubGlobal('fetch',vi.fn(async()=>new Response(JSON.stringify({message:'not a schema'}))));render(<Documentation param="reference/train"/>);expect(await screen.findByRole('alert')).toBeTruthy();expect(screen.getByRole('link',{name:'Read the guides →'})).toBeTruthy();
 vi.stubGlobal('fetch',vi.fn(async()=>new Response(JSON.stringify(schema))));fireEvent.click(screen.getByRole('button',{name:'Retry schema'}));expect(await screen.findByText('Live Training reference')).toBeTruthy();
});
it('shows not-found state and dismisses mobile navigation',()=>{
 render(<Documentation param="guides/missing"/>);expect(screen.getByRole('heading',{name:'Documentation page not found'})).toBeTruthy();fireEvent.click(screen.getByRole('button',{name:'Toggle documentation navigation'}));expect(screen.getByRole('button',{name:'Close documentation navigation'})).toBeTruthy();fireEvent.keyDown(window,{key:'Escape'});expect(screen.queryByRole('button',{name:'Close documentation navigation'})).toBeNull();
});

it('keeps the header simple with a GitHub repository link',()=>{
 render(<Documentation param="guides/quickstart"/>);
 expect(screen.getByText('Developer docs')).toBeTruthy();
 expect(screen.queryByRole('navigation',{name:'Documentation sections'})).toBeNull();
 expect(screen.getByRole('link',{name:'GitHub: wenkesj/trains'}).getAttribute('href')).toBe('https://github.com/wenkesj/trains');
});
it('keeps development navigation open while Python can be toggled',async()=>{
 render(<Documentation param="guides/overview"/>);
 const python=screen.getByRole('button',{name:'Python'});
 expect(python.getAttribute('aria-expanded')).toBe('false');
 expect(screen.getByRole('group',{name:'Training API sections'})).toBeTruthy();
 fireEvent.click(python);
 expect(await screen.findByRole('group',{name:'Python code sections'})).toBeTruthy();
 fireEvent.click(python);
 expect(screen.queryByRole('group',{name:'Python code sections'})).toBeNull();
 expect(screen.getByRole('group',{name:'Training API sections'})).toBeTruthy();
});
it('embeds Python objects in the shared page and opens their navigation on deep links',async()=>{
 render(<Documentation param="reference/python/trains.datasets.chat/trains.datasets.chat.chat_dataset_from_records"/>);
 expect(await screen.findByText('Materialize complete conversations.')).toBeTruthy();
 expect(screen.getByRole('button',{name:'Python'}).getAttribute('aria-expanded')).toBe('true');
 expect(screen.getByRole('group',{name:'Training API sections'})).toBeTruthy();
 expect(document.querySelector('a[href="#/docs/guides/overview"]')).toBeTruthy();
 expect(screen.getByRole('article',{name:'trains.datasets.chat code reference'}).querySelector('details')).toBeTruthy();
 expect(document.querySelector('iframe')).toBeNull();
});
it('keeps guide navigation when Training is selected and follows route changes',async()=>{
 const {rerender}=render(<Documentation param="reference/train/operation/post-api-v1-create-session"/>);
 expect(await screen.findByText('Live Training reference')).toBeTruthy();
 expect(document.querySelector('a[href="#/docs/guides/overview"]')).toBeTruthy();
 expect(screen.getByRole('group',{name:'Training API sections'})).toBeTruthy();
 rerender(<Documentation param="reference/python"/>);
 expect(screen.getByRole('button',{name:'Python'}).getAttribute('aria-expanded')).toBe('true');
 expect(screen.getByRole('group',{name:'Training API sections'})).toBeTruthy();
 rerender(<Documentation param="guides/quickstart"/>);
 expect(screen.getByRole('button',{name:'Python'}).getAttribute('aria-expanded')).toBe('false');
});
it('searches generated Python symbols with in-app links',async()=>{
 render(<Documentation param="guides/overview"/>);
 await screen.findByRole('button',{name:'Python'});
 fireEvent.click(screen.getByRole('button',{name:/Search docs/}));
 fireEvent.change(screen.getByRole('textbox',{name:'Search documentation'}),{target:{value:'chat_dataset_from_records'}});
 const title=await within(screen.getByRole('dialog',{name:'Search documentation'})).findByText(pythonModule.objects[0].id);
 expect(title.closest('a')?.getAttribute('href')).toBe('#/docs/reference/python/trains.datasets.chat/trains.datasets.chat.chat_dataset_from_records');
});
