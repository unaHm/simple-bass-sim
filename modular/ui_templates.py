"""
ui_templates.py - Full UI template (HTML string) with 'Save Compressor (global)' button.
"""

HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8"><title>Bass Simulator (Modular)</title>
<style>
  body { font-family: Arial, sans-serif; background:#111;color:#eee;padding:18px; }
  .controls { display:flex; gap:18px; }
  .col { flex:1; min-width:260px; }
  table{width:100%;border-collapse:collapse}
  th,td{padding:6px;border:1px solid #333;text-align:center}
  input[type=range]{width:100%}
  .num-input{width:70px}
  .pickup-select{width:120px}
  .meter{height:14px;background:#222;border:1px solid #333;border-radius:4px;overflow:hidden}
  .meter>span{display:block;height:100%;background:#4caf50;width:0%}
  .small{font-size:0.85rem;color:#bbb}
  button{padding:6px 8px;margin:4px 0}
  .disabled { opacity:0.5; }
</style>
</head>
<body>
<h1>Bass Simulator</h1>
<div class="controls">
  <div class="col">
    <h3>Presets</h3>
    <input id="preset_name" placeholder="Preset name" style="width:100%"><br><br>
    <button onclick="savePreset()">Save Preset</button>
    <button onclick="deletePreset()">Delete</button><br><br>
    <label>Factory</label>
    <select id="factory_list" size="6" style="width:100%"></select>
    <button onclick="loadFactory()">Load Factory</button><br><br>
    <label>User</label>
    <select id="preset_list" size="6" style="width:100%"></select>
    <button onclick="loadPreset()">Load User</button>
  </div>

  <div class="col">
    <h3>Pickup Editor (mm)</h3>
    <div class="small">Distance from bridge to closest pickup edge. One value per pickup (applies to all strings).</div>
    <table id="pickup_table">
      <thead><tr><th>String</th><th>P1 Type</th><th>P1 Closest (mm)</th><th>P2 Type</th><th>P2 Closest (mm)</th></thead>
      <tbody></tbody>
    </table>
    <div style="margin-top:8px;">
      <label style="font-weight:bold;">
        <input type="checkbox" id="enable_pickup_2"> Enable Pickup 2
      </label>
    </div>
    <div class="small" style="margin-top:6px">Tip: changing any P1 slider updates pickup 1 for all strings.</div>
  </div>

  <div class="col">
    <h3>Meters & Effects</h3>
    <label>Peak: <span id="peak">-120</span>dB</label><div class="meter"><span id="peak_bar"></span></div>
    <label>Comp GR: <span id="comp_gr">0</span>dB</label><div class="meter"><span id="comp_bar"></span></div>
    <label>Limiter GR: <span id="lim_gr">0</span>dB</label><div class="meter"><span id="lim_bar"></span></div>

    <h4>Pickup & Master</h4>
    <label>P1 Vol <input id="p1" type="range" min="0" max="1" step="0.01"></label>
    <label>P2 Vol <input id="p2" type="range" min="0" max="1" step="0.01"></label>
    <label>Master <input id="mg" type="range" min="0" max="2" step="0.01"></label>

    <h4>Effects</h4>
    <div>
      <label>SVF Cutoff <input id="svf_base_cutoff" type="range" min="100" max="5000"></label>
      <label><input id="svf_bypass" type="checkbox"> Bypass</label>
    </div>
    <div>
      <label>SVF Env Depth <input id="svf_env_depth" type="range" min="0" max="5000"></label>
    </div>
    <div>
      <label>Oct Dry <input id="oct_dry" type="range" min="0" max="1" step="0.01"></label>
      <label><input id="oct_bypass" type="checkbox"> Bypass</label>
    </div>
    <div>
      <label>Oct Sub Gain <input id="oct_sub_gain" type="range" min="0" max="2" step="0.01"></label>
    </div>
    <div>
      <label>Comp Threshold <input id="comp_threshold" type="range" min="-40" max="0" step="1"></label>
      <label><input id="comp_bypass" type="checkbox"> Bypass</label>
    </div>
    <div>
      <label>Comp Ratio <input id="comp_ratio" type="range" min="1" max="20" step="0.1"></label>
      <label>Comp Makeup <input id="comp_makeup" type="range" min="0" max="4" step="0.01"></label>
    </div>

    <div style="margin-top:10px;">
      <button onclick="saveGlobalCompressor()">Save Compressor (global)</button>
      <div class="small">Click to persist compressor params/bypass across reboots.</div>
    </div>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.6.1/socket.io.min.js"></script>
<script>
const socket = io();
socket.on('meter_data', d=>{
  document.getElementById('peak').textContent = (d.peak_db||-120).toFixed(1);
  document.getElementById('peak_bar').style.width = Math.min(100, Math.max(0,(d.peak_db+60)/60*100)) + '%';
  document.getElementById('comp_gr').textContent = (d.comp_gr_db||0).toFixed(2);
  document.getElementById('comp_bar').style.width = Math.min(100, Math.abs(d.comp_gr_db||0)/10*100)+'%';
  document.getElementById('lim_gr').textContent = (d.limiter_gr_db||0).toFixed(2);
  document.getElementById('lim_bar').style.width = Math.min(100, Math.abs(d.limiter_gr_db||0)/3*100)+'%';
});

function postJSON(url, body){ return fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}).then(r=>r.json()); }
function fetchJSON(url){ return fetch(url).then(r=>r.json()); }

function refreshPresetLists(){
  fetchJSON('/api/list_presets').then(d=>{
    const f = document.getElementById('factory_list'), u = document.getElementById('preset_list');
    f.innerHTML=''; u.innerHTML='';
    (d.factory||[]).forEach(p=>{ const o=document.createElement('option'); o.value=p; o.textContent=p; f.appendChild(o); });
    (d.user||[]).forEach(p=>{ const o=document.createElement('option'); o.value=p; o.textContent=p; u.appendChild(o); });
  });
}

function savePreset(){ const name=document.getElementById('preset_name').value.trim(); if(!name){alert('enter name');return;} postJSON('/api/save_preset',{name}).then(r=>{ if(r.status==='ok'){ refreshPresetLists(); alert('saved'); } else alert('save failed: '+JSON.stringify(r)); }); }

function loadPreset(){
  const sel = document.getElementById('preset_list'); const name = sel.value; if(!name){ alert('select a preset'); return; }
  postJSON('/api/load_preset',{name}).then(r=>{
    if(r.status==='ok' && r.instrument_state){ applyInstrumentState(r.instrument_state); } else fetchInstrumentState();
  });
}

function loadFactory(){
  const sel = document.getElementById('factory_list'); const name = sel.value; if(!name){ alert('select'); return; }
  postJSON('/api/load_preset',{name}).then(r=>{ if(r.status==='ok' && r.instrument_state){ applyInstrumentState(r.instrument_state); } else fetchInstrumentState(); });
}

function deletePreset(){ const sel=document.getElementById('preset_list'), name=sel.value; if(!name){alert('select');return;} if(!confirm('delete?'))return; postJSON('/api/delete_preset',{name}).then(r=>{ if(r.status==='ok') refreshPresetLists(); else alert('delete failed'); }); }

function buildPickupTable(state){
  const tbody = document.querySelector('#pickup_table tbody'); tbody.innerHTML='';
  const names = state.string_names || ['B','E','A','D','G','C'];
  const per_pickup = state.closest_distance_mm_per_pickup || [40,40];
  const ptypes = state.pickup_types || ['splitP','none'];
  for(let i=0;i<names.length;i++){
    const tr=document.createElement('tr');
    const tdName=document.createElement('td'); tdName.textContent=names[i]; tr.appendChild(tdName);

    const tdP1Type=document.createElement('td');
    const selP1=document.createElement('select'); selP1.className='pickup-select';
    ['single','splitP','humbucker','soapbar','none'].forEach(opt=>{ const o=document.createElement('option'); o.value=opt; o.textContent=opt; if(opt===ptypes[0]) o.selected=true; selP1.appendChild(o); });
    if(i===0){
      selP1.id = 'pickup1_type';
      selP1.onchange=function(e){ postJSON('/api/set_pickup_type',{pickup_slot:0,type:e.target.value}).then(()=>fetchInstrumentState()); };
    } else {
      selP1.onchange = null;
    }
    tdP1Type.appendChild(selP1); tr.appendChild(tdP1Type);

    const tdP1Dist=document.createElement('td');
    const r1=document.createElement('input'); r1.type='range'; r1.min=0; r1.max=200; r1.step=0.1; r1.value=per_pickup[0];
    const n1=document.createElement('input'); n1.type='number'; n1.step=0.1; n1.className='num-input'; n1.value=per_pickup[0];
    if(i===0){ r1.id='pickup1_closest_mm_slider'; n1.id='pickup1_closest_mm_number'; }
    r1.oninput=function(e){ n1.value=e.target.value; postPickupDistanceAndRefresh(0, parseFloat(e.target.value)); };
    n1.onchange=function(e){ r1.value=e.target.value; postPickupDistanceAndRefresh(0, parseFloat(e.target.value)); };
    tdP1Dist.appendChild(r1); tdP1Dist.appendChild(document.createTextNode(' ')); tdP1Dist.appendChild(n1); tr.appendChild(tdP1Dist);

    const tdP2Type=document.createElement('td');
    const selP2=document.createElement('select'); selP2.className='pickup-select';
    ['single','splitP','humbucker','soapbar','none'].forEach(opt=>{ const o=document.createElement('option'); o.value=opt; o.textContent=opt; if(opt===ptypes[1]) o.selected=true; selP2.appendChild(o); });
    if(i===0){
      selP2.id = 'pickup2_type';
      selP2.onchange=function(e){ postJSON('/api/set_pickup_type',{pickup_slot:1,type:e.target.value}).then(()=>fetchInstrumentState()); };
    } else {
      selP2.onchange = null;
    }
    tdP2Type.appendChild(selP2); tr.appendChild(tdP2Type);

    const tdP2Dist=document.createElement('td');
    const r2=document.createElement('input'); r2.type='range'; r2.min=0; r2.max=200; r2.step=0.1; r2.value=per_pickup[1];
    const n2=document.createElement('input'); n2.type='number'; n2.step=0.1; n2.className='num-input'; n2.value=per_pickup[1];
    if(i===0){ r2.id='pickup2_closest_mm_slider'; n2.id='pickup2_closest_mm_number'; }
    r2.oninput=function(e){ n2.value=e.target.value; postPickupDistanceAndRefresh(1, parseFloat(e.target.value)); };
    n2.onchange=function(e){ r2.value=e.target.value; postPickupDistanceAndRefresh(1, parseFloat(e.target.value)); };
    tdP2Dist.appendChild(r2); tdP2Dist.appendChild(document.createTextNode(' ')); tdP2Dist.appendChild(n2); tr.appendChild(tdP2Dist);

    tbody.appendChild(tr);
  }
}

function postPickupDistanceAndRefresh(pickup_slot, mm){
  postJSON('/api/set_pickup_distance',{pickup_slot: pickup_slot, distance_mm: mm}).then(r=>{
    if(r.status==='ok'){
      fetchInstrumentState();
    } else {
      console.warn('set pickup distance failed', r);
    }
  }).catch(e=>console.warn('error posting pickup distance', e));
}

function setPickup2EnabledUI(enabled){
  const checkbox = document.getElementById('enable_pickup_2');
  const t2 = document.getElementById('pickup2_type');
  const s2 = document.getElementById('pickup2_closest_mm_slider');
  const n2 = document.getElementById('pickup2_closest_mm_number');
  const p2vol = document.getElementById('p2');

  checkbox.checked = !!enabled;
  if(!t2 || !s2 || !n2 || !p2vol) return;

  if(enabled){
    t2.disabled = false; s2.disabled = false; n2.disabled = false; p2vol.disabled = false;
    t2.classList.remove('disabled'); s2.classList.remove('disabled'); n2.classList.remove('disabled');
  } else {
    t2.disabled = true; s2.disabled = true; n2.disabled = true; p2vol.disabled = true;
    t2.classList.add('disabled'); s2.classList.add('disabled'); n2.classList.add('disabled');
  }
}

document.addEventListener('DOMContentLoaded', ()=>{
  const enableCheckbox = document.getElementById('enable_pickup_2');
  enableCheckbox.addEventListener('change', function(e){
    const enabled = e.target.checked;
    postJSON('/api/set_pickup_enabled', { enabled: enabled }).then(r=>{
      if(r.status === 'ok'){
        setPickup2EnabledUI(enabled);
        fetchInstrumentState();
      } else {
        console.warn('set_pickup_enabled failed', r);
      }
    }).catch(err=>console.warn('set_pickup_enabled error', err));
  });

  document.getElementById('p1').addEventListener('input', e=>postJSON('/api/update_controls',{pickup1_volume: parseFloat(e.target.value)}));
  document.getElementById('p2').addEventListener('input', e=>postJSON('/api/update_controls',{pickup2_volume: parseFloat(e.target.value)}));
  document.getElementById('mg').addEventListener('input', e=>postJSON('/api/update_controls',{master_gain: parseFloat(e.target.value)}));

  ['svf_base_cutoff','svf_env_depth','oct_dry','oct_sub_gain','comp_threshold','comp_ratio','comp_makeup'].forEach(id=>{
    const el=document.getElementById(id); if(!el) return;
    el.addEventListener('input', e=> postJSON('/api/update_effects_params',{param:id, value: parseFloat(e.target.value)}));
  });

  document.getElementById('svf_bypass').addEventListener('change', e=>postJSON('/api/set_bypass',{name:'env_filter', state: e.target.checked}));
  document.getElementById('oct_bypass').addEventListener('change', e=>postJSON('/api/set_bypass',{name:'octaver', state: e.target.checked}));
  document.getElementById('comp_bypass').addEventListener('change', e=>postJSON('/api/set_bypass',{name:'comp', state: e.target.checked}));

  document.getElementById('comp_bypass').addEventListener('change', ()=>{
    // don't persist automatically; user must click Save Compressor to persist
  });

  refreshPresetLists();
  fetchInstrumentState();
});

// apply instrument state to UI (called after load_preset)
function applyInstrumentState(state){
  const pickup_state = {
    closest_distance_mm_per_pickup: state.closest_distance_mm_per_pickup || [40,40],
    pickup_types: state.pickup_types || ['splitP','none'],
    string_names: ['B','E','A','D','G','C']
  };
  buildPickupTable(pickup_state);

  try{
    const per = pickup_state.closest_distance_mm_per_pickup;
    const s1 = document.getElementById('pickup1_closest_mm_slider'); const n1 = document.getElementById('pickup1_closest_mm_number');
    const s2 = document.getElementById('pickup2_closest_mm_slider'); const n2 = document.getElementById('pickup2_closest_mm_number');
    const t1 = document.getElementById('pickup1_type'); const t2 = document.getElementById('pickup2_type');
    if(s1) s1.value = per[0]; if(n1) n1.value = per[0];
    if(s2) s2.value = per[1]; if(n2) n2.value = per[1];
    if(t1) t1.value = pickup_state.pickup_types[0]; if(t2) t2.value = pickup_state.pickup_types[1];
    setPickup2EnabledUI(!(pickup_state.pickup_types[1] === 'none'));
  }catch(e){ console.warn('applyInstrumentState id update failed', e); }

  fetchJSON('/api/get_state').then(s=>{
    try{ const dsp = s.dsp_params || []; if(dsp.length>=4){ document.getElementById('p1').value=dsp[0]; document.getElementById('p2').value=dsp[1]; document.getElementById('mg').value=dsp[3]; } }catch(e){}
    if(s.effects){
      ['svf_base_cutoff','svf_env_depth','oct_dry','oct_sub_gain','comp_threshold','comp_ratio','comp_makeup'].forEach(k=>{
        const el=document.getElementById(k); if(el){ el.value = s.effects[k] || el.value; }
      });
    }
    if(s.effects_bypass){
      document.getElementById('svf_bypass').checked = !!s.effects_bypass['env_filter'];
      document.getElementById('oct_bypass').checked = !!s.effects_bypass['octaver'];
      // show comp bypass but remember compressor persistence is explicit
      document.getElementById('comp_bypass').checked = !!s.effects_bypass['comp'];
    }
  });
}

function fetchInstrumentState(){
  fetchJSON('/api/get_instrument_params').then(s=>{
    const pickup_state = { closest_distance_mm_per_pickup: s.closest_distance_mm_per_pickup || [40,40], pickup_types: s.pickup_types || ['splitP','none'], string_names: ['B','E','A','D','G','C'] };
    buildPickupTable(pickup_state);
    try{
      const per = pickup_state.closest_distance_mm_per_pickup;
      const s1 = document.getElementById('pickup1_closest_mm_slider'); const n1 = document.getElementById('pickup1_closest_mm_number');
      const s2 = document.getElementById('pickup2_closest_mm_slider'); const n2 = document.getElementById('pickup2_closest_mm_number');
      const t1 = document.getElementById('pickup1_type'); const t2 = document.getElementById('pickup2_type');
      if(s1) s1.value = per[0]; if(n1) n1.value = per[0];
      if(s2) s2.value = per[1]; if(n2) n2.value = per[1];
      if(t1) t1.value = pickup_state.pickup_types[0]; if(t2) t2.value = pickup_state.pickup_types[1];
      setPickup2EnabledUI(!(pickup_state.pickup_types[1] === 'none'));
    }catch(e){}
  });
  fetchJSON('/api/get_state').then(s=>{
    try{ const dsp = s.dsp_params || []; if(dsp.length>=4){ document.getElementById('p1').value=dsp[0]; document.getElementById('p2').value=dsp[1]; document.getElementById('mg').value=dsp[3]; } }catch(e){}
    if(s.effects){
      ['svf_base_cutoff','svf_env_depth','oct_dry','oct_sub_gain','comp_threshold','comp_ratio','comp_makeup'].forEach(k=>{
        const el=document.getElementById(k); if(el){ el.value = s.effects[k] || el.value; }
      });
    }
    if(s.effects_bypass){
      document.getElementById('svf_bypass').checked = !!s.effects_bypass['env_filter'];
      document.getElementById('oct_bypass').checked = !!s.effects_bypass['octaver'];
      document.getElementById('comp_bypass').checked = !!s.effects_bypass['comp'];
    }
  });
}

// Save compressor global (explicit)
function saveGlobalCompressor(){
  const bypass = document.getElementById('comp_bypass').checked;
  const thresh = parseFloat(document.getElementById('comp_threshold').value);
  const ratio = parseFloat(document.getElementById('comp_ratio').value);
  const makeup = parseFloat(document.getElementById('comp_makeup').value);
  postJSON('/api/set_comp_state', { bypass: bypass, comp_params: { comp_threshold: thresh, comp_ratio: ratio, comp_makeup: makeup } })
    .then(r => {
      if(r && r.status === 'ok'){
        alert('Global compressor saved.');
      } else {
        alert('Save failed: ' + JSON.stringify(r));
      }
    }).catch(e => alert('Save failed: '+e));
}
</script>
</body>
</html>
"""
