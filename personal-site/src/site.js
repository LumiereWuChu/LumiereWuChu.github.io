const locale=document.body.dataset.locale||'zh';
try{localStorage.setItem('lumiere-language',locale)}catch{}
const toggle=document.querySelector('.menu-toggle'),nav=document.querySelector('#main-nav');
toggle?.addEventListener('click',()=>{const open=toggle.getAttribute('aria-expanded')!=='true';toggle.setAttribute('aria-expanded',String(open));nav.classList.toggle('open',open);if(open)nav.querySelector('a')?.focus()});
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&toggle&&toggle.getAttribute('aria-expanded')==='true'){toggle.setAttribute('aria-expanded','false');nav.classList.remove('open');toggle.focus()}});
let toastTimer;const toast=s=>{const el=document.querySelector('.toast');el.textContent=s;el.classList.add('show');clearTimeout(toastTimer);toastTimer=setTimeout(()=>el.classList.remove('show'),2400)};
document.querySelectorAll('[data-copy]').forEach(button=>button.addEventListener('click',async()=>{try{await navigator.clipboard.writeText(button.dataset.copy);toast(locale==='zh'?'引用已复制':'Citation copied')}catch{window.prompt(locale==='zh'?'复制引用':'Copy citation',button.dataset.copy)}}));
let filter='all';const input=document.querySelector('#search');
function updateSearch(){const query=(input?.value||'').toLocaleLowerCase().trim();let count=0;document.querySelectorAll('[data-entry]').forEach(e=>{const show=(filter==='all'||e.dataset.kind===filter)&&e.dataset.search.includes(query);e.hidden=!show;if(show)count++});const countEl=document.querySelector('#result-count');if(countEl)countEl.textContent=count+(locale==='zh'?' 篇':' entries');const no=document.querySelector('#no-results');if(no)no.hidden=count>0;}
input?.addEventListener('input',updateSearch);
document.querySelectorAll('[data-filter]').forEach(b=>b.addEventListener('click',()=>{filter=b.dataset.filter;document.querySelectorAll('[data-filter]').forEach(a=>a.setAttribute('aria-pressed',String(a===b)));updateSearch()}));
document.querySelector('#print-cv')?.addEventListener('click',()=>window.print());
const dialog=document.querySelector('#photo-dialog');
document.querySelectorAll('[data-photo]').forEach(b=>b.addEventListener('click',()=>{const im=dialog.querySelector('img');im.src=b.dataset.photo;im.alt=b.dataset.alt;dialog.showModal()}));
dialog?.querySelector('.dialog-close')?.addEventListener('click',()=>dialog.close());
dialog?.addEventListener('click',e=>{if(e.target===dialog)dialog.close()});
