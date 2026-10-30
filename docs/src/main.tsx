import React, {useEffect, useState} from 'react';
import {createRoot} from 'react-dom/client';
import Documentation from './Documentation';
import './style.css';

function App() {
  const [hash,setHash]=useState(location.hash);
  useEffect(()=>{const change=()=>setHash(location.hash);window.addEventListener('hashchange',change);return()=>window.removeEventListener('hashchange',change);},[]);
  const parts=hash.replace(/^#\/?/,'').split('/');
  return <Documentation param={parts[0]==='docs'?parts.slice(1).join('/')||null:null}/>;
}
createRoot(document.getElementById('root')!).render(<React.StrictMode><App/></React.StrictMode>);
