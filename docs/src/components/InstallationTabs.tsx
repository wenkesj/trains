import {useId, useRef, useState} from "react";

export default function InstallationTabs({commands}: {commands: string[]}) {
  const [selected, setSelected] = useState(0);
  const id = useId();
  const tabs = useRef<(HTMLButtonElement | null)[]>([]);
  return <div className="docs-code docs-installation">
    <div className="docs-installation-tabs" role="tablist" aria-label="Installation method">
      {commands.map((command, index) => {
        const tool = command.split(" ")[0];
        return <button key={tool} ref={element=>{tabs.current[index]=element;}} id={`${id}-tab-${index}`} type="button" role="tab" aria-selected={selected===index} aria-controls={`${id}-panel-${index}`} tabIndex={selected===index ? 0 : -1} onClick={()=>setSelected(index)} onKeyDown={event=>{
          let next: number;
          if (event.key === "ArrowRight") next = (index + 1) % commands.length;
          else if (event.key === "ArrowLeft") next = (index + commands.length - 1) % commands.length;
          else if (event.key === "Home") next = 0;
          else if (event.key === "End") next = commands.length - 1;
          else return;
          event.preventDefault();setSelected(next);tabs.current[next]?.focus();
        }}>{tool === "poetry" ? "Poetry" : tool}</button>;
      })}
    </div>
    {commands.map((command, index)=><div key={command} role="tabpanel" id={`${id}-panel-${index}`} aria-labelledby={`${id}-tab-${index}`} hidden={selected!==index} tabIndex={0}><pre><code className="language-sh">{command}</code></pre></div>)}
  </div>;
}
