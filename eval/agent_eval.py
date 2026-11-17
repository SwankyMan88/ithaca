"""Coding-agent test bench: the same tasks and the same user prompts, different system instructions ("variants"),
several runs each, scored by hidden checks. Talks to a running Ithaca (python eval/agent_eval.py --help).

  python eval/agent_eval.py --variants baseline,short,xml --runs 3
  python eval/agent_eval.py --tasks fix_bugs,docx --variants baseline --runs 1 --verbose

The "baseline" prompt and tool list are read from the Coding page itself (src/ui.cpp, through node), so it always
matches what the app ships. Every run gets a fresh copy of its project in %TEMP%\\ithaca_eval. Results: a JSONL log,
full transcripts and a Markdown summary in eval/results/."""
import argparse, json, os, re, shutil, subprocess, sys, tempfile, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tasks as T
import variants as V

ROOT = os.path.dirname(HERE)


def post(port, path, body, timeout=900):
    req = urllib.request.Request(f'http://127.0.0.1:{port}{path}', data=json.dumps(body).encode(), method='POST', headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode('utf8', 'replace'))


def get(port, path):
    with urllib.request.urlopen(f'http://127.0.0.1:{port}{path}', timeout=60) as r:
        return json.loads(r.read().decode('utf8', 'replace'))


def chat(port, messages, temperature, max_tokens):
    """Streams /api/chat; returns (text, stats, error)."""
    body = json.dumps({'messages': messages, 'temperature': temperature, 'max_tokens': max_tokens, 'stop': ['</tool_call>'], 'json_after': os.environ.get('ITHACA_JSON_AFTER', '<tool_call>')}).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{port}/api/chat', data=body, method='POST')
    text, stats, err, buf = '', None, None, ''
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            buf += raw.decode('utf8', 'replace')
            while '\n\n' in buf:
                line, buf = buf.split('\n\n', 1)
                if not line.startswith('data: '):
                    continue
                ev = json.loads(line[6:])
                text += ev.get('t', '')
                if ev.get('error'):
                    err = ev['error']
                if ev.get('done'):
                    stats = ev.get('stats')
    return text, stats, err


# ---- the page's prompt and tools, straight from ui.cpp ----
def page_source():
    s = open(os.path.join(ROOT, 'src', 'ui.cpp'), encoding='utf8').read()
    a = s.index('const CODE_TOOLS=['); b = s.index('];', a) + 2
    c = s.index('function cwSystem(){'); d = s.index('function cwShow(', c)
    return s[a:b], s[c:d]


def baseline(name):
    tools_js, sys_js = page_source()
    js = tools_js + '\nlet cw={name:' + json.dumps(name) + '},cs={items:[],msgs:[]};\n' + sys_js + '\nprocess.stdout.write(JSON.stringify({prompt:cwSystem(),tools:CODE_TOOLS}));'
    p = subprocess.run(['node', '-e', js], capture_output=True, text=True, encoding='utf8')
    if p.returncode:
        raise SystemExit('could not read the Coding page prompt with node:\n' + p.stderr[-800:])
    return json.loads(p.stdout)


# ---- tool-call parsing: the same forms the page accepts ----
def think_split(t):
    m = re.search(r'<think>([\s\S]*?)(</think>|$)', t)
    return (m.group(1), t[:m.start()] + t[m.end():]) if m else ('', t)


def loads_loose(s):
    """JSON as models write it: raw newlines/tabs inside strings are allowed (same as cwJson in ui.cpp)."""
    try:
        return json.loads(s, strict=False)
    except ValueError:
        pass
    s = re.sub(r'"""([\s\S]*?)"""', lambda m: json.dumps(m.group(1)), s)   # Python triple quotes
    try:
        return json.loads(s, strict=False)
    except ValueError:
        pass
    o, q, j = [], False, 0   # a stray \" outside a string ("new": \"x\") is a quote
    while j < len(s):
        c = s[j]
        if q:
            if c == '\\' and s[j + 1:j + 2] == '"' and re.match(r'\s*[,}\]:]', s[j + 2:j + 12]):
                o.append('"'); q = False; j += 2; continue   # a stray \" closing the string
            if c == '\\':
                o.append(s[j:j + 2]); j += 2; continue
            if c == '"': q = False
        elif c == '\\' and s[j + 1:j + 2] == '"':
            j += 1; continue
        elif c == '"': q = True
        o.append(c); j += 1
    return json.loads(''.join(o), strict=False)


def parse_call(text, tool_names):
    text = re.sub(r'<(/?)tools>', r'<\1tool_call>', text)
    i = text.rfind('<tool_call>')
    if i >= 0:
        s = re.sub(r'</tool_call>[\s\S]*$', '', text[i + 11:]).strip()
        s = re.sub(r'^```(?:json)?\s*', '', s); s = re.sub(r'```\s*$', '', s)
        fm = re.match(r'^<function=([\w.-]+)>([\s\S]*?)(?:</function>|$)', s)
        if fm:
            args = {k: v for k, v in re.findall(r'<parameter=([\w.-]+)>\n?([\s\S]*?)\n?</parameter>', fm.group(2))}
            for k in ('depth', 'start_line', 'end_line', 'timeout'):
                if k in args:
                    try: args[k] = int(args[k])
                    except ValueError: pass
            if 'replace_all' in args:
                args['replace_all'] = args['replace_all'].strip().lower() == 'true'
            return {'name': fm.group(1), 'args': args}
        try:
            j = loads_loose(s)
            return {'name': j.get('name'), 'args': j.get('arguments') or j.get('parameters') or {}}
        except Exception:
            return parse_call(s, tool_names) or {'bad': s[:300]}   # several calls in one block: the first valid one
    body = think_split(text)[1]
    for m in re.finditer(r'\{\s*"name"\s*:', body):   # bare / fenced JSON naming one of the tools
        depth, q, esc, k = 0, False, False, m.start()
        for j in range(k, len(body)):
            ch = body[j]
            if q:
                if esc: esc = False
                elif ch == '\\': esc = True
                elif ch == '"': q = False
                continue
            if ch == '"': q = True
            elif ch == '{': depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    try:
                        o = loads_loose(body[k:j + 1])
                        if o.get('name') in tool_names:
                            return {'name': o['name'], 'args': o.get('arguments') or o.get('parameters') or {}}
                    except Exception:
                        pass
                    break
    return None


def visible(body):
    body = re.sub(r'<tool_call>[\s\S]*$', '', body)
    body = re.sub(r'```(?:json)?\s*\{\s*"name"\s*:[\s\S]*$', '', body)
    return re.sub(r'\{\s*"name"\s*:\s*"[a-z_]+"[\s\S]*$', '', body).strip()


NUDGE = re.compile(r"\b(I will|I'll|I am going to|Let me|Let's|Now,? I|Next,? I)\b", re.I)
CHANGES = {'edit_file', 'write_file', 'create_file', 'set_cell', 'make_dir', 'rename', 'delete'}


def run_agent(port, ws, variant, prompt, tool_names, max_steps, log):
    msgs = [{'role': 'user', 'content': prompt}]
    m = dict(steps=0, calls=0, tool_errors=0, bad_calls=0, nudges=0, gen_tokens=0, prompt_tokens=0, changes=0, commands=0)
    final = ''
    for step in range(max_steps):
        seen = 0   # the page's trimming: only the 6 newest tool outputs stay in full
        for x in reversed(msgs):
            if x['role'] == 'user' and x['content'].startswith('<tool_response>'):
                seen += 1
                if seen > 6 and len(x['content']) > 400:
                    x['content'] = '<tool_response>\n(older output removed to save space; read the file again if needed)\n</tool_response>'
        text, stats, err = chat(port, [{'role': 'system', 'content': variant['system']}] + msgs, variant['temperature'], variant.get('max_tokens', 2048))
        m['steps'] += 1
        if stats:
            m['gen_tokens'] += stats.get('gen', 0); m['prompt_tokens'] += stats.get('prompt', 0) - stats.get('reused', 0)
        log.append({'role': 'assistant', 'text': text, 'error': err})
        if err:
            m['error'] = err; break
        call = parse_call(text, tool_names)
        msgs.append({'role': 'assistant', 'content': text + ('\n</tool_call>' if call else '')})
        if not call:
            body = think_split(text)[1]
            if step > 0 and m['nudges'] < 2 and NUDGE.search(visible(body)[-400:]):
                m['nudges'] += 1
                msgs.append({'role': 'user', 'content': 'Go ahead and do that now with a tool call. If everything is already done, give the summary instead.'})
                continue
            final = visible(body); break
        if 'bad' in call:
            m['bad_calls'] += 1
            out = 'Error: the tool call was not valid JSON. Write it as {"name": ..., "arguments": {...}}. File content goes in one ordinary JSON string: write new lines as \\n and escape double quotes as \\", and never use triple quotes (""").'
        else:
            m['calls'] += 1
            m['changes'] += call['name'] in CHANGES; m['commands'] += call['name'] == 'run_command'
            r = post(port, '/api/code/tool', {'root': ws, 'name': call['name'], 'args': call['args']})
            ok = bool(r.get('ok')) and not r.get('error')
            if not ok:
                m['tool_errors'] += 1
            out = r.get('error') or r.get('output') or ''
            if len(out) > 14000:
                out = out[:14000] + '\n... (cut: read a smaller part)'
            out = ('' if ok else 'Error: ') + (out or 'done')
            log.append({'role': 'tool', 'name': call['name'], 'args': call['args'], 'ok': ok, 'output': out[:4000]})
        # loop guard, as on the page: the same call with the same result is flagged, the third repeat stops the task
        sig = json.dumps([call.get('name'), call.get('args', call.get('bad')), out])
        again = sig == m.get('_last'); m['_last'] = sig
        if not again: m['_run'] = 0
        if again or (sig in m.setdefault('_seen', set()) and call.get('name') not in ('read_file', 'list_dir', 'search')):
            m['loops'] = m.get('loops', 0) + 1
            if again: m['_run'] = m.get('_run', 0) + 1
            if m.get('_run', 0) >= 2:
                m['stopped_loop'] = True; m.pop('_seen'); break
            out += '\n\n(Note from Ithaca: you already made this exact call and got this exact result. Do not repeat it. Try a different fix, or stop and explain what is blocking you.)'
        m['_seen'].add(sig)
        msgs.append({'role': 'user', 'content': f'<tool_response>\n{out}\n</tool_response>'})
    else:
        m['hit_step_limit'] = True
    for k in ('_seen', '_last', '_run'): m.pop(k, None)
    return final, m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--variants', default='baseline', help='comma list from eval/variants.py: ' + ', '.join(V.VARIANTS))
    ap.add_argument('--tasks', default='all', help='comma list: ' + ', '.join(t['id'] for t in T.TASKS))
    ap.add_argument('--runs', type=int, default=3, help='runs per task and variant (models are random: 3+ gives a real rate)')
    ap.add_argument('--port', type=int, default=7860)
    ap.add_argument('--model', help='load this GGUF first (otherwise the loaded model is used)')
    ap.add_argument('--ctx', type=int, default=16384)
    ap.add_argument('--max-steps', type=int, default=25)
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args()
    if a.model:
        r = post(a.port, '/api/load', {'path': a.model, 'ctx': a.ctx})
        if not r.get('loaded'):
            raise SystemExit('load failed: ' + json.dumps(r)[:300])
    st = get(a.port, '/api/status')
    if not st.get('loaded'):
        raise SystemExit('no model loaded in Ithaca (use --model)')
    model = st.get('name') or os.path.basename(st.get('model', ''))
    if st.get('ctx', 0) < 8192:
        print(f'warning: the model has a {st.get("ctx")}-token context; 16K is recommended (--model ... --ctx 16384)')
    variants = [v.strip() for v in a.variants.split(',') if v.strip()]
    for v in variants:
        if v not in V.VARIANTS: raise SystemExit(f'unknown variant {v}; have: {", ".join(V.VARIANTS)}')
    tasks = T.TASKS if a.tasks == 'all' else [t for t in T.TASKS if t['id'] in a.tasks.split(',')]
    stamp = time.strftime('%Y%m%d-%H%M%S')
    outdir = os.path.join(HERE, 'results', stamp)
    os.makedirs(os.path.join(outdir, 'transcripts'), exist_ok=True)
    base_ws = os.path.join(tempfile.gettempdir(), 'ithaca_eval')
    rows = []
    total = len(variants) * len(tasks) * a.runs
    print(f'{model}: {len(tasks)} tasks x {len(variants)} variants x {a.runs} runs = {total} runs -> {outdir}')
    n = 0
    for run in range(a.runs):          # interleaved, so a slow patch on the PC doesn't hit one variant only
        for t in tasks:
            for vname in variants:
                n += 1
                ws = os.path.join(base_ws, 'ws')
                shutil.rmtree(ws, ignore_errors=True); os.makedirs(ws)
                t['setup'](ws)
                before = T.snapshot(ws)
                base = baseline(os.path.basename(ws))
                variant = V.VARIANTS[vname](base, os.path.basename(ws))
                tool_names = {x[0] for x in base['tools']}
                log = [{'role': 'system', 'text': variant['system']}, {'role': 'user', 'text': t['prompt']}]
                t0 = time.time()
                try:
                    final, m = run_agent(a.port, ws, variant, t['prompt'], tool_names, a.max_steps, log)
                    passed, note = (T.chk_question(ws, final, before) if t['check'] is None else t['check'](ws, final))
                except Exception as e:
                    final, m, passed, note = '', {'error': repr(e)}, False, 'runner error: ' + repr(e)[:200]
                m['seconds'] = round(time.time() - t0, 1)
                row = dict(variant=vname, task=t['id'], run=run + 1, passed=bool(passed), note=note, final=final[:600], **m)
                rows.append(row)
                with open(os.path.join(outdir, 'results.jsonl'), 'a', encoding='utf8') as f:
                    f.write(json.dumps(row) + '\n')
                with open(os.path.join(outdir, 'transcripts', f'{t["id"]}-{vname}-{run + 1}.json'), 'w', encoding='utf8') as f:
                    json.dump(log, f, indent=1)
                print(f'[{n}/{total}] {vname:12} {t["id"]:9} run {run + 1}: {"PASS" if passed else "fail"}  {m.get("steps", 0)} steps, {m["seconds"]} s  {note[:90]}')
                if a.verbose:
                    for e in log[2:]:
                        print('   ', (e.get('name') or e['role']), json.dumps(e.get('args') or e.get('text', ''))[:160])
    write_summary(outdir, model, rows, variants, tasks, a)


def write_summary(outdir, model, rows, variants, tasks, a):
    def agg(rs):
        k = len(rs) or 1
        return dict(rate=sum(r['passed'] for r in rs) / k, steps=sum(r.get('steps', 0) for r in rs) / k, secs=sum(r.get('seconds', 0) for r in rs) / k,
                    errs=sum(r.get('tool_errors', 0) + r.get('bad_calls', 0) for r in rs) / k, tok=sum(r.get('gen_tokens', 0) for r in rs) / k)
    L = [f'# Coding agent eval: {model}', '', f'{len(tasks)} tasks, {a.runs} runs each, max {a.max_steps} steps. Pass = the hidden check succeeded.', '',
         '| Variant | Pass rate | Avg steps | Avg seconds | Tool errors / run | Tokens written / run |', '|---|---|---|---|---|---|']
    for v in variants:
        s = agg([r for r in rows if r['variant'] == v])
        L.append(f'| {v} | {s["rate"]:.0%} | {s["steps"]:.1f} | {s["secs"]:.0f} | {s["errs"]:.1f} | {s["tok"]:.0f} |')
    L += ['', '## Pass rate by task', '', '| Task | ' + ' | '.join(variants) + ' |', '|---|' + '---|' * len(variants)]
    for t in tasks:
        L.append(f'| {t["id"]} | ' + ' | '.join(f'{agg([r for r in rows if r["variant"] == v and r["task"] == t["id"]])["rate"]:.0%}' for v in variants) + ' |')
    L += ['', '## Failures', '']
    for r in rows:
        if not r['passed']:
            L.append(f'- {r["variant"]} / {r["task"]} / run {r["run"]}: {r["note"]} ({r.get("steps", 0)} steps)')
    open(os.path.join(outdir, 'summary.md'), 'w', encoding='utf8').write('\n'.join(L) + '\n')
    print('\n'.join(L[:8 + len(variants)]))
    print('\nfull summary:', os.path.join(outdir, 'summary.md'))


if __name__ == '__main__':
    main()
