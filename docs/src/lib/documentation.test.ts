import {expect,it} from 'vitest';
import {API_SOURCES,GUIDES,guideOutline,documentationIndex} from './documentation';
it('indexes only public pages',()=>{expect(API_SOURCES).toEqual([]);expect(GUIDES.some(page=>['training','inference'].includes(page.id))).toBe(false);expect(documentationIndex([]).length).toBe(17);});
it('ignores fenced headings',()=>{expect(guideOutline('## First\n```bash\n## Comment\n```\n## Next\n')).toHaveLength(2);});
