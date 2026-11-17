/* Proposal Pipeline — prototype UI logic (vanilla JS, no dependencies). */
'use strict';

const EXAMPLE = {
  title: 'Senior Full-Stack Engineer — Geospatial Analytics Platform',
  company: 'GeoInsight Analytics',
  description:
    'We need an end-to-end geospatial analytics dashboard that ingests municipal parcel and zoning datasets, runs spatial joins and proximity analysis on a PostGIS backend, and renders interactive choropleth maps. The existing system is a Django monolith with slow ad-hoc SQL, no caching, and 8+ second query times on large parcels.',
  tech_stack: 'Next.js, Node.js, TypeScript, PostgreSQL, PostGIS, Redis',
  domain: 'Geospatial Analytics',
  budget: '$90k–$120k',
  timeline: '16 weeks',
};

const $ = (id) => document.getElementById(id);

const byName = (name) => document.querySelector(`[name="${name}"]`);

function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

function splitList(value) {
  return (value || '').split(',').map((s) => s.trim()).filter(Boolean);
}

function section(title, bodyHtml) {
  return `<div class="section"><h3>${esc(title)}</h3>${bodyHtml}</div>`;
}

function ul(items) {
  if (!items || !items.length) return '';
  return `<ul>${items.map((i) => `<li>${esc(i)}</li>`).join('')}</ul>`;
}

function chip(label, value) {
  return `<span class="chip"><span class="chip__k">${esc(label)}</span>${esc(value)}</span>`;
}

async function loadStatus() {
  const text = $('status-text');
  const dot = document.querySelector('.status__dot');
  try {
    const res = await fetch('/health');
    if (!res.ok) throw new Error('unhealthy');
    const data = await res.json();
    text.textContent = `${data.model} · ${data.hardware_backend} · ${data.inference_backend}`;
    dot.classList.add('is-ok');
  } catch (_err) {
    text.textContent = 'unreachable';
  }
}

function setLoading(loading) {
  const button = $('submit');
  button.disabled = loading;
  button.textContent = loading ? 'Drafting…' : 'Generate proposal';
}

function renderResult(data) {
  const p = data.proposal;
  const milestonesHtml = (p.milestones || [])
    .map((m) =>
      `<li><span class="m-name">${esc(m.name)}</span>` +
      `<span class="m-meta">${m.duration_weeks}w</span>` +
      `<span class="m-focus">${esc(m.focus)}</span></li>`
    )
    .join('');

  const bodyHtml =
    section('Executive hook', `<p>${esc(p.executive_hook)}</p>`) +
    section('Proposed architecture', ul(p.architecture)) +
    section('Milestones', `<ol class="milestones">${milestonesHtml}</ol>`) +
    section('Timeline & deliverables', `<p class="kicker">Timeline — ${esc(p.timeline)}</p>` + ul(p.deliverables));

  $('output').innerHTML = bodyHtml;

  const meta = $('meta');
  meta.hidden = false;
  meta.innerHTML =
    chip('backend', data.backend) +
    chip('latency', `${data.latency_ms} ms`) +
    chip('tokens', String(data.token_usage.total_tokens)) +
    chip('adapter', data.lora_adapter_version);
}

function renderError(message) {
  const box = document.createElement('p');
  box.className = 'error';
  box.textContent = `Could not generate a proposal — ${message}`;
  $('output').replaceChildren(box);
  $('meta').hidden = true;
}

function fillForm(values) {
  byName('title').value = values.title || '';
  byName('company').value = values.company || '';
  byName('description').value = values.description || '';
  byName('tech_stack').value = values.tech_stack || '';
  byName('domain').value = values.domain || '';
  byName('timeline').value = values.timeline || '';
  byName('budget').value = values.budget || '';
}

async function handleSubmit(event) {
  event.preventDefault();
  const payload = {
    title: byName('title').value.trim(),
    company: byName('company').value.trim(),
    description: byName('description').value.trim(),
    tech_stack: splitList(byName('tech_stack').value),
    domain: byName('domain').value.trim(),
    timeline: byName('timeline').value.trim(),
    budget: byName('budget').value.trim(),
  };

  setLoading(true);
  try {
    const res = await fetch('/api/v1/generate-proposal', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    renderResult(data);
  } catch (err) {
    renderError(err.message);
  } finally {
    setLoading(false);
  }
}

function resetForm() {
  byName('title').value = '';
  byName('company').value = '';
  byName('description').value = '';
  byName('tech_stack').value = '';
  byName('domain').value = '';
  byName('timeline').value = '';
  byName('budget').value = '';
}

$('job-form').addEventListener('submit', handleSubmit);
$('load-example').addEventListener('click', () => fillForm(EXAMPLE));
$('clear').addEventListener('click', resetForm);

loadStatus();