"""
ui_templates.py - Full UI template (HTML string) with 'Save Compressor (global)' button.
"""

HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8"><title>Bass Simulator (Modular)</title>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap" rel="stylesheet">
<style>
  /* Base Styles & Orbitron Font */
  body { 
    font-family: 'Orbitron', sans-serif; 
    background:#000;
    color:#00ff00;
    padding:15px; 
    font-size: 18px; /* Base font size */
  }
  h1 { 
    font-size: 4rem; /* H1: 4rem */
    color: #00ff00; 
    text-shadow: 0 0 8px #00ff00; 
    border-bottom: 2px solid #004400; 
    padding-bottom: 10px; 
    margin-bottom: 20px;
  }
  h3 {
    font-size: 2.5rem; /* H3: 2.5rem */
    color: #00ff00; 
    text-shadow: 0 0 5px #00ff00; 
    border-bottom: 1px solid #004400; 
    padding-bottom: 8px; 
    margin-top: 20px;
  }
  h4 {
    font-size: 1.8rem; /* H4: 1.8rem */
    color: #00ff00; 
    border-bottom: 1px solid #001100;
    padding-top: 10px;
    padding-bottom: 5px;
  }
  
  /* Always single column (3 rows) for mobile display */
  .controls { display:flex; flex-direction: column; gap:30px; } 
  .col { 
    background:#080808; 
    padding:25px; 
    border:2px solid #004400; 
    box-shadow: 0 0 15px rgba(0,255,0,0.4); 
    border-radius:6px; 
    min-width: unset;
  }

  /* Tables and Inputs */
  table{width:100%;border-collapse:collapse; margin-top: 25px;}
  th,td{padding:12px;border:1px solid #004400;text-align:center; font-size:1.2rem;} 
  
  /* Slider group for numerical display */
  input[type=range]{
    width:calc(100% - 200px); 
    vertical-align: middle; 
    margin-right: 15px;
    height: 30px; /* Taller slider track for easier tap */
  }
  .slider-group { display: flex; align-items: center; margin-bottom: 20px; } 
  .slider-label { flex-basis: 150px; font-size: 2.5rem; } /* Very Large Label font */
  .slider-value { flex-basis: 100px; text-align: right; font-size: 4rem; color: #00ff00; } /* HUGE Value font (4rem) */
  
  .num-input{width:100px; background: #111; border: 2px solid #00ff00; color: #00ff00; padding: 15px; font-size: 1.5rem;} 
  .pickup-select{width:100%; background: #111; border: 2px solid #00ff00; color: #00ff00; padding: 10px; font-size: 1.5rem;}
  
  /* Meters */
  .meter-label { font-size: 2rem; margin-top: 10px; display: block;}
  .meter{height:25px;background:#111;border:2px solid #00ff00;border-radius:4px;overflow:hidden; margin-bottom: 15px;}
  .meter>span{display:block;height:100%;background:#00ff00;width:0%; transition: width 0.1s linear;}
  
  /* Presets */
  #factory_list, #preset_list { background: #111; border: 2px solid #00ff00; color: #00ff00; font-size: 1.5rem;}

  .small{font-size:1.2rem;color:#999} 
  button{padding:15px 20px;margin:8px 0; background:#004400; color:#00ff00; border:2px solid #00ff00; cursor:pointer; font-family: 'Orbitron', sans-serif; font-size: 1.5rem;} 
  button:hover { background:#006600; box-shadow: 0 0 10px #00ff00; }
  .disabled { opacity:0.5; }

  /* Hidden, Glowing Checkboxes (Bypass Buttons) */
  .checkbox-group { display: flex; flex-wrap: wrap; justify-content: space-around; margin-bottom: 20px; gap: 10px;}
  .checkbox-container { flex-grow: 1; margin: 0; }
  .checkbox-container input[type=checkbox] { 
    position: absolute; 
    opacity: 0; 
    width: 0; 
    height: 0; 
  }
  .checkbox-container label {
    display: block; /* Make label fill its flex container */
    text-align: center;
    padding: 20px 10px; /* Adjust padding for multi-word labels */
    background: #111;
    color: #999;
    border: 2px solid #004400;
    cursor: pointer;
    transition: all 0.2s;
    font-size: 2.5rem; /* Large button text (2.5rem) */
    font-weight: 700;
  }
  /* The default state is UNCHECKED (bypassed/off). We style CHECKED to be ON/active. */
  .checkbox-container input[type=checkbox]:checked + label {
    color: #00ff00;
    border-color: #00ff00;
    box-shadow: 0 0 12px #00ff00; 
    background: #002200;
  }
  .checkbox-container label:hover {
    color: #00ff00;
  }
  
</style>
</head>
<body>
<h1>Bass Simulator</h1>
<div class="controls">

  <div class="col">
    <h3>Meters</h3>
    
    <label class="meter-label">Peak: <span id="peak">-120</span>dB</label><div class="meter"><span id="peak_bar"></span></div>
    <label class="meter-label">Limiter GR: <span id="lim_gr">0</span>dB</label><div class="meter"><span id="lim_bar"></span></div>

    <h3>Effect Activation Controls</h3>
    <div class="checkbox-group">
        <div class="checkbox-container">
            <input id="svf_bypass" type="checkbox">
            <label for="svf_bypass">SVF</label>
        </div>
        <div class="checkbox-container">
            <input id="oct_bypass" type="checkbox">
            <label for="oct_bypass">Octaver</label>
        </div>
        <div class="checkbox-container">
            <input id="comp_bypass" type="checkbox">
            <label for="comp_bypass">Compressor</label>
        </div>
    </div>

    <h4>Pickup & Master</h4>
    <div class="slider-group"><label class="slider-label">P1 Vol</label><input id="p1" type="range" min="0" max="1" step="0.01"><span id="p1_value" class="slider-value">0</span></div>
    <div class="slider-group"><label class="slider-label">P2 Vol</label><input id="p2" type="range" min="0" max="1" step="0.01"><span id="p2_value" class="slider-value">0</span></div>
    <div class="slider-group"><label class="slider-label">Master</label><input id="mg" type="range" min="0" max="2" step="0.01"><span id="mg_value" class="slider-value">0</span></div>

    <h4>State Variable Filter (SVF)</h4>
    <div class="slider-group"><label class="slider-label">Cutoff</label><input id="svf_base_cutoff" type="range" min="100" max="5000"><span id="svf_base_cutoff_value" class="slider-value">100</span></div>
    <div class="slider-group"><label class="slider-label">Env Depth</label><input id="svf_env_depth" type="range" min="0" max="5000"><span id="svf_env_depth_value" class="slider-value">0</span></div>
    
    <h4>Octaver</h4>
    <div class="slider-group"><label class="slider-label">Dry/Sub Mix</label><input id="oct_mix" type="range" min="0" max="1" step="0.01"><span id="oct_mix_value" class="slider-value">0</span></div>
    
    <h4>Compressor</h4>
    <label class="meter-label">Comp GR: <span id="comp_gr">0</span>dB</label><div class="meter"><span id="comp_bar"></span></div>
    <div class="slider-group"><label class="slider-label">Threshold</label><input id="comp_threshold" type="range" min="-40" max="0" step="1"><span id="comp_threshold_value" class="slider-value">-40</span></div>
    <div class="slider-group"><label class="slider-label">Ratio</label><input id="comp_ratio" type="range" min="1" max="20" step="0.1"><span id="comp_ratio_value" class="slider-value">1</span></div>
    <div class="slider-group"><label class="slider-label">Makeup</label><input id="comp_makeup" type="range" min="0" max="4" step="0.01"><span id="comp_makeup_value" class="slider-value">0</span></div>

    <div style="margin-top:30px;">
      <button onclick="saveGlobalCompressor()">Save Compressor (global)</button>
      <div class="small">Click to persist compressor params/activation across reboots.</div>
    </div>
  </div>

  <div class="col">
    <h3>Presets</h3>
    
    <label>Factory</label>
    <select id="factory_list" size="6" style="width:100%"></select>
    <button onclick="loadFactory()">Load Factory</button><br><br>
    
    <label>User</label>
    <select id="preset_list" size="6" style="width:100%"></select>
    <button onclick="loadPreset()">Load User</button><br><br>
    
    <hr style="border-color: #004400; margin: 20px 0;">
    
    <input id="preset_name" placeholder="Preset name" style="width:100%; background: #111; border: 2px solid #00ff00; color: #00ff00; padding: 15px; font-size: 1.5rem;"><br><br>
    <button onclick="savePreset()">Save Preset</button>
    <button onclick="deletePreset()">Delete</button>
  </div>


  <div class="col">
    <h3>Pickup Editor (mm)</h3>
    <div class="small">Distance from bridge to closest pickup edge. One value per pickup (applies to all strings).</div>
    <table id="pickup_table">
      <thead><tr><th>String</th><th>P1 Type</th><th>P1 Closest (mm)</th><th>P2 Type</th><th>P2 Closest (mm)</th></tr></thead>
      <tbody></tbody>
    </table>
    <div style="margin-top:25px;" class="checkbox-container">
      <input type="checkbox" id="enable_pickup_2">
      <label for="enable_pickup_2" style="font-weight:bold; display: inline-block;">Enable Pickup 2</label>
    </div>
    <div class="small" style="margin-top:12px">Tip: changing any P1 slider updates pickup 1 for all strings.</div>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.6.1/socket.io.min.js"></script>
<script>
const socket = io();

// Array of slider IDs and their corresponding value span IDs (for numerical display)
const SLIDER_MAP = [
    { id: 'p1', decimals: 2 },
    { id: 'p2', decimals: 2 },
    { id: 'mg', decimals: 2 },
    { id: 'svf_base_cutoff', decimals: 0 },
    { id: 'svf_env_depth', decimals: 0 },
    { id: 'oct_mix', decimals: 2 }, // New Octave Mix fader
    { id: 'comp_threshold', decimals: 0 },
    { id: 'comp_ratio', decimals: 1 },
    { id: 'comp_makeup', decimals: 2 },
];

function updateSliderValueDisplay(sliderId, value) {
    // This is for the numeric spans in the Meters & Effects column
    const slider = SLIDER_MAP.find(s => s.id === sliderId);
    if (!slider) return;
    const valueEl = document.getElementById(sliderId + '_value');
    if (valueEl) {
        // Ensure to handle distance sliders which have different IDs but share the update logic for initial state
        if (sliderId.includes('closest_mm')) {
            valueEl.textContent = parseFloat(value).toFixed(1);
        } else {
            valueEl.textContent = parseFloat(value).toFixed(slider.decimals);
        }
    }
}

// Function to attach input listeners for sliders (both control and numerical display)
function setupSlider(id, updateFunc, decimals) {
    const el = document.getElementById(id);
    if (!el) return;
    
    // Initial display update
    updateSliderValueDisplay(id, el.value);

    // Event listener for input (updates value display and sends API call)
    el.addEventListener('input', e => {
        const val = parseFloat(e.target.value);
        updateSliderValueDisplay(id, val);
        updateFunc(val);
    });
}

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

function savePreset(){ const name=document.getElementById('preset_name').value.trim(); if(!name){alert('enter name');return;} postJSON('/api/save_preset',{name}).then(r=>{ if(r.status==='ok'){ refreshPresetLists(); alert('saved'); } else alert('save failed: '+JSON.stringify(r)); });
}

function loadPreset(){
  const sel = document.getElementById('preset_list'); const name = sel.value; if(!name){ alert('select a preset'); return;
  }
  postJSON('/api/load_preset',{name}).then(r=>{
    if(r.status==='ok' && r.instrument_state){ applyInstrumentState(r.instrument_state); } else fetchInstrumentState();
  });
}

function loadFactory(){
  const sel = document.getElementById('factory_list'); const name = sel.value; if(!name){ alert('select'); return;
  }
  postJSON('/api/load_preset',{name}).then(r=>{ if(r.status==='ok' && r.instrument_state){ applyInstrumentState(r.instrument_state); } else fetchInstrumentState(); });
}

function deletePreset(){ const sel=document.getElementById('preset_list'), name=sel.value; if(!name){alert('select');return;} if(!confirm('delete?'))return;
postJSON('/api/delete_preset',{name}).then(r=>{ if(r.status==='ok') refreshPresetLists(); else alert('delete failed'); }); }

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
      selP1.onchange=function(e){ postJSON('/api/set_pickup_type',{pickup_slot:0,type:e.target.value}).then(()=>fetchInstrumentState());
      };
    } else {
      selP1.onchange = null;
    }
    tdP1Type.appendChild(selP1); tr.appendChild(tdP1Type);

    const tdP1Dist=document.createElement('td');
    // Note: Pickup distance uses a range input (r1) AND a number input (n1) which serves as the numerical display.
    const r1=document.createElement('input'); r1.type='range'; r1.min=0; r1.max=200; r1.step=0.1; r1.value=per_pickup[0];
    const n1=document.createElement('input'); n1.type='number'; n1.step=0.1; n1.className='num-input'; n1.value=per_pickup[0];
    if(i===0){ r1.id='pickup1_closest_mm_slider'; n1.id='pickup1_closest_mm_number';
    }
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
    if(i===0){ r2.id='pickup2_closest_mm_slider'; n2.id='pickup2_closest_mm_number';
    }
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
  const p2val = document.getElementById('p2_value'); // Added element for display

  checkbox.checked = !!enabled;
  if(!t2 || !s2 || !n2 || !p2vol) return;
  
  if(enabled){
    t2.disabled = false; s2.disabled = false; n2.disabled = false; p2vol.disabled = false;
    t2.classList.remove('disabled'); s2.classList.remove('disabled'); n2.classList.remove('disabled');
    if(p2val) p2val.classList.remove('disabled');
  } else {
    t2.disabled = true; s2.disabled = true; n2.disabled = true; p2vol.disabled = true;
    t2.classList.add('disabled'); s2.classList.add('disabled');
    n2.classList.add('disabled');
    if(p2val) p2val.classList.add('disabled');
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

  // Setup listeners for Pickup/Master controls (using the new setupSlider function)
  setupSlider('p1', (val)=>postJSON('/api/update_controls',{pickup1_volume: val}), 2);
  setupSlider('p2', (val)=>postJSON('/api/update_controls',{pickup2_volume: val}), 2);
  setupSlider('mg', (val)=>postJSON('/api/update_controls',{master_gain: val}), 2);

  // Setup listener for Octaver Mix control
  setupSlider('oct_mix', (val) => {
    // Octave Mix Logic: dry = 1 - mix, sub_gain = mix
    const dry = (1 - val).toFixed(2);
    const sub_gain = val.toFixed(2);

    postJSON('/api/update_effects_params', { param: 'oct_dry', value: parseFloat(dry) });
    postJSON('/api/update_effects_params', { param: 'oct_sub_gain', value: parseFloat(sub_gain) });
  }, 2);
  
  // Setup listeners for other Effects controls
  ['svf_base_cutoff','svf_env_depth','comp_threshold','comp_ratio','comp_makeup'].forEach(id=>{
    const slider = SLIDER_MAP.find(s => s.id === id);
    if (slider) {
        setupSlider(id, (val)=>postJSON('/api/update_effects_params',{param:id, value: val}), slider.decimals);
    }
  });

  // INVERTED LOGIC: checked (true) means BYPASS IS OFF (false) and effect is ACTIVE
  document.getElementById('svf_bypass').addEventListener('change', e=>postJSON('/api/set_bypass',{name:'env_filter', state: !e.target.checked}));
  document.getElementById('oct_bypass').addEventListener('change', e=>postJSON('/api/set_bypass',{name:'octaver', state: !e.target.checked}));
  document.getElementById('comp_bypass').addEventListener('change', e=>postJSON('/api/set_bypass',{name:'comp', state: !e.target.checked}));

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
    const s2 = document.getElementById('pickup2_closest_mm_slider');
    const n2 = document.getElementById('pickup2_closest_mm_number');
    const t1 = document.getElementById('pickup1_type'); const t2 = document.getElementById('pickup2_type');
    if(s1) s1.value = per[0]; 
    if(n1) n1.value = per[0];
    if(s2) s2.value = per[1]; 
    if(n2) n2.value = per[1];
    if(t1) t1.value = pickup_state.pickup_types[0]; if(t2) t2.value = pickup_state.pickup_types[1];
    setPickup2EnabledUI(!(pickup_state.pickup_types[1] === 'none'));
  }catch(e){ console.warn('applyInstrumentState id update failed', e); }

  fetchJSON('/api/get_state').then(s=>{
    try{ const dsp = s.dsp_params || []; if(dsp.length>=4){ 
        document.getElementById('p1').value=dsp[0]; updateSliderValueDisplay('p1', dsp[0]);
        document.getElementById('p2').value=dsp[1]; updateSliderValueDisplay('p2', dsp[1]);
        document.getElementById('mg').value=dsp[3]; updateSliderValueDisplay('mg', dsp[3]);
     } }catch(e){}
    if(s.effects){
      ['svf_base_cutoff','svf_env_depth','comp_threshold','comp_ratio','comp_makeup'].forEach(k=>{
        const el=document.getElementById(k); 
        if(el){ 
            el.value = s.effects[k] || el.value; 
            updateSliderValueDisplay(k, el.value);
        }
      });
      
      // Handle Octave Mix
      const oct_sub_gain = s.effects['oct_sub_gain'] || 0; // The oct_mix value is equal to the sub gain value
      const oct_mix_el = document.getElementById('oct_mix');
      if (oct_mix_el) {
          oct_mix_el.value = oct_sub_gain;
          updateSliderValueDisplay('oct_mix', oct_sub_gain);
      }
    }
    if(s.effects_bypass){
      // INVERTED LOGIC: API returns TRUE if BYPASSED, so we check the box if it is FALSE/ACTIVE
      document.getElementById('svf_bypass').checked = !s.effects_bypass['env_filter'];
      document.getElementById('oct_bypass').checked = !s.effects_bypass['octaver'];
      document.getElementById('comp_bypass').checked = !s.effects_bypass['comp'];
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
      if(s1) s1.value = per[0]; 
      if(n1) n1.value = per[0];
      if(s2) s2.value = per[1]; 
      if(n2) n2.value = per[1];
      if(t1) t1.value = pickup_state.pickup_types[0]; if(t2) t2.value = pickup_state.pickup_types[1];
      setPickup2EnabledUI(!(pickup_state.pickup_types[1] === 'none'));
    }catch(e){}
  });
  fetchJSON('/api/get_state').then(s=>{
    try{ const dsp = s.dsp_params || []; if(dsp.length>=4){ 
        document.getElementById('p1').value=dsp[0]; updateSliderValueDisplay('p1', dsp[0]);
        document.getElementById('p2').value=dsp[1]; updateSliderValueDisplay('p2', dsp[1]);
        document.getElementById('mg').value=dsp[3]; updateSliderValueDisplay('mg', dsp[3]);
     } }catch(e){}
    if(s.effects){
      ['svf_base_cutoff','svf_env_depth','comp_threshold','comp_ratio','comp_makeup'].forEach(k=>{
        const el=document.getElementById(k); 
        if(el){ 
            el.value = s.effects[k] || el.value;
            updateSliderValueDisplay(k, el.value); 
        }
      });
      // Handle Octave Mix
      const oct_sub_gain = s.effects['oct_sub_gain'] || 0; // The oct_mix value is equal to the sub gain value
      const oct_mix_el = document.getElementById('oct_mix');
      if (oct_mix_el) {
          oct_mix_el.value = oct_sub_gain;
          updateSliderValueDisplay('oct_mix', oct_sub_gain);
      }
    }
    if(s.effects_bypass){
      // INVERTED LOGIC: API returns TRUE if BYPASSED, so we check the box if it is FALSE/ACTIVE
      document.getElementById('svf_bypass').checked = !s.effects_bypass['env_filter'];
      document.getElementById('oct_bypass').checked = !s.effects_bypass['octaver'];
      document.getElementById('comp_bypass').checked = !s.effects_bypass['comp'];
    }
  });
}

// Save compressor global (explicit)
function saveGlobalCompressor(){
  // INVERTED LOGIC: bypass state sent to API is the opposite of the checkbox state
  const bypass = !document.getElementById('comp_bypass').checked;
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
