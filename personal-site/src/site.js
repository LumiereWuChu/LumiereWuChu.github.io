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
if(dialog){
 const buttons=[...document.querySelectorAll('[data-photo]')];
 const image=dialog.querySelector('img');
 const status=dialog.querySelector('.lightbox-status');
 const previous=dialog.querySelector('[data-photo-prev]');
 const next=dialog.querySelector('[data-photo-next]');
 let group=[],index=0,opener=null,request=0,previousOverflow='';
 function showPhoto(position){
  index=(position+group.length)%group.length;
  const button=group[index],currentRequest=++request;
  image.hidden=true;image.alt=button.dataset.alt||'';
  status.textContent=dialog.dataset.loading;
  dialog.querySelector('#lightbox-title').textContent=button.dataset.photoTitle||button.getAttribute('aria-label')||'';
  dialog.querySelector('#lightbox-caption').textContent=button.dataset.caption||'';
  dialog.querySelector('#lightbox-count').textContent=`${index+1} / ${group.length}`;
  dialog.querySelector('.lightbox-original').href=button.dataset.photo;
  previous.disabled=next.disabled=group.length<2;
  const pendingImage=new Image();
  pendingImage.onload=()=>{if(currentRequest!==request||!dialog.open)return;image.src=pendingImage.src;image.hidden=false;status.textContent='';};
  pendingImage.onerror=()=>{if(currentRequest===request&&dialog.open)status.textContent=dialog.dataset.error;};
  pendingImage.src=button.dataset.photo;
 }
 buttons.forEach(button=>button.addEventListener('click',()=>{
  opener=button;
  group=buttons.filter(item=>(item.dataset.gallery||'entries')===(button.dataset.gallery||'entries'));
  previousOverflow=document.body.style.overflow;document.body.style.overflow='hidden';
  dialog.showModal();showPhoto(group.indexOf(button));
  dialog.querySelector('.dialog-close').focus();
 }));
 previous.addEventListener('click',()=>showPhoto(index-1));
 next.addEventListener('click',()=>showPhoto(index+1));
 dialog.querySelector('.dialog-close').addEventListener('click',()=>dialog.close());
 dialog.addEventListener('keydown',event=>{
  if(event.altKey||event.ctrlKey||event.metaKey)return;
  if(event.key==='ArrowLeft'){event.preventDefault();showPhoto(index-1);}
  if(event.key==='ArrowRight'){event.preventDefault();showPhoto(index+1);}
 });
 dialog.addEventListener('close',()=>{request++;document.body.style.overflow=previousOverflow;opener?.focus();});
 dialog.addEventListener('click',event=>{if(event.target===dialog)dialog.close();});
}
