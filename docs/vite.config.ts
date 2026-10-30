import {defineConfig} from 'vitest/config';
import react from '@vitejs/plugin-react';
export default defineConfig(({mode})=>({plugins:[react()],define:{__DEVELOPMENT_DOCS__:JSON.stringify(mode === 'development-docs')},test:{environment:'jsdom',include:mode === 'development-docs' ? ['src/**/*.development.test.{ts,tsx}'] : ['src/**/*.test.{ts,tsx}'],exclude:mode === 'development-docs' ? [] : ['src/**/*.development.test.{ts,tsx}']}}));
