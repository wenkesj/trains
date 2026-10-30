import {cleanup, render} from '@testing-library/react';
import {afterEach,expect,it,vi} from 'vitest';
import ApiDocumentation from './ApiDocumentation';
import {API_SOURCES,type LoadedApi} from '../lib/documentation';
const reference=vi.hoisted(()=>({configuration:{} as Record<string,unknown>}));
vi.mock('@scalar/api-reference-react',()=>({ApiReferenceReact:({configuration}:{configuration:Record<string,unknown>})=>{reference.configuration=configuration;return <div>Reference</div>;}}));
const api:LoadedApi={source:API_SOURCES[0],error:null,document:{openapi:'3.1.0',info:{title:'Trains',version:'1'},paths:{'/api/v1/create_session':{get:{summary:'List sessions'},post:{summary:'Create session'}},'/traces':{get:{summary:'List traces'}}}}};
afterEach(cleanup);
it('focuses the selected operation and keeps Scalar links inside the app',()=>{
 render(<ApiDocumentation api={api} focus="docs/reference/train/operation/post-api-v1-create-session/description/introduction"/>);expect(reference.configuration.content).toMatchObject({paths:{'/api/v1/create_session':{post:{summary:'Create session'}}}});expect(Object.keys((reference.configuration.content as {paths:object}).paths)).toEqual(['/api/v1/create_session']);
 const redirect=reference.configuration.redirect as (path:string)=>string;expect(redirect('#/docs/reference/train/api-1/tag/api/v1/create_session/docs/reference/train/operation/post-api-v1-create-session')).toBe('#/docs/reference/train/operation/post-api-v1-create-session');expect(reference.configuration.hideTestRequestButton).toBe(true);expect(JSON.stringify(reference.configuration.content)).toContain('trains');expect(JSON.stringify(reference.configuration.content)).not.toContain('robo');
});
