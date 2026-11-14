"""Chat test bench: the same questions under different system instructions, scored by automatic checks.

  python eval/chat_eval.py --variants baseline,short,plain --runs 3
  python eval/chat_eval.py --model C:/models/gemma-3-4b-it-Q4_K_M.gguf --runs 2

"baseline" is the Chat area's own system prompt, read from src/ui.cpp through node (as a new user with no saved
memories would get it). Every question has a check that needs no judgement, so the scores are reproducible.
Results go to eval/results/chat-<time>/."""
import argparse, json, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from agent_eval import post, get   # same HTTP helpers

ROOT = os.path.dirname(HERE)


def page_prompt():
    s = open(os.path.join(ROOT, 'src', 'ui.cpp'), encoding='utf8').read()
    a = s.index('function systemPrompt(extra,lastUser){'); b = s.index('  return s}', a) + len('  return s}')
    js = ('let prefs={memory:true},memories=[],chat=null;const fileContext=()=>"",searchMem=()=>[],memLine=m=>m.text;\n' + s[a:b] +
          '\nprocess.stdout.write(systemPrompt("",""));')
    p = subprocess.run(['node', '-e', js], capture_output=True, text=True, encoding='utf8')
    if p.returncode:
        raise SystemExit('could not read the Chat prompt with node:\n' + p.stderr[-600:])
    return p.stdout


def chat(port, system, user, temperature, max_tokens=1024):
    body = json.dumps({'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}], 'temperature': temperature, 'max_tokens': max_tokens}).encode()
    import urllib.request
    req = urllib.request.Request(f'http://127.0.0.1:{port}/api/chat', data=body, method='POST')
    text, stats, buf = '', None, ''
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            buf += raw.decode('utf8', 'replace')
            while '\n\n' in buf:
                line, buf = buf.split('\n\n', 1)
                if line.startswith('data: '):
                    ev = json.loads(line[6:]); text += ev.get('t', '')
                    if ev.get('done'): stats = ev.get('stats')
    return text, stats


def body_of(t):
    """What the user would see: no thinking, no hidden tags."""
    t = re.sub(r'<think>[\s\S]*?(</think>|$)', '', t)
    return re.sub(r'<(memory_\w+|generate_\w+)[^>]*?(/>|>[\s\S]*?</\1>)', '', t).strip()


def words(t): return len(re.findall(r"[\w'-]+", t))
def sentences(t): return len([s for s in re.split(r'(?<=[.!?])\s+', t.strip()) if re.search(r'\w', s)])


def code_block(t, lang='python'):
    m = re.search(r'```(?:' + lang + r'|py)?\s*\n([\s\S]*?)```', t)
    return m.group(1) if m else None


def run_py(code, test):
    p = subprocess.run([sys.executable, '-I', '-c', code + '\n' + test], capture_output=True, text=True, timeout=30)
    return p.returncode == 0, (p.stderr or p.stdout)[-200:]


def chk_palindrome(t, raw):
    c = code_block(t)
    if not c:
        return False, 'no python block'
    return run_py(c, 'assert is_palindrome("Never odd or even"); assert is_palindrome("Racecar"); assert not is_palindrome("hello")')


def chk_json(t, raw):
    s = re.sub(r'^```(?:json)?\s*|\s*```$', '', t.strip())
    try:
        j = json.loads(s)
    except Exception:
        return False, 'not pure JSON: ' + t[:80]
    return (j == {'name': 'Ana', 'age': 30}), json.dumps(j)[:80]


def has(*pats):
    return lambda t, raw: (any(re.search(p, t, re.I) for p in pats), '' if any(re.search(p, t, re.I) for p in pats) else 'missing: ' + pats[0])


QUESTIONS = [
    dict(id='math', q='What is 17 times 24?', check=has(r'\b408\b')),
    dict(id='trick', q='A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?', check=has(r'\$?0?\.05\b', r'\b5 cents\b', r'five cents')),
    dict(id='one_sentence', q='In one sentence, what is photosynthesis?', check=lambda t, raw: (sentences(t) <= 1 and words(t) <= 45 and re.search(r'light|sun', t, re.I) is not None, f'{sentences(t)} sentences, {words(t)} words')),
    dict(id='bullets', q='List three primary colors as a bulleted list and nothing else.', check=lambda t, raw: (len(re.findall(r'^\s*[-*•]\s+\S', t, re.M)) == 3 and len([l for l in t.splitlines() if l.strip()]) == 3, t[:80].replace('\n', ' | '))),
    dict(id='json_only', q='Give me a JSON object with the keys name and age for a 30-year-old named Ana. Reply with only the JSON.', check=chk_json),
    dict(id='code', q='Write a Python function is_palindrome(s) that ignores case and spaces.', check=chk_palindrome),
    dict(id='unknowable', q='What did I have for breakfast today?', check=has(r"(don'?t|do not|can'?t|cannot|no way to|have no|not able to|unable to) (know|tell|see|access|have)", r"no (way|information|access)")),
    dict(id='offline', q="What's the weather in Paris right now?", check=has(r'(internet|real-?time|live|current) (access|data|information|weather)', r"(can'?t|cannot|don'?t have|no) (access|check|browse|look up)", r'offline')),
    dict(id='french', q='Réponds en français : quelle est la capitale du Japon ?', check=lambda t, raw: (re.search(r'Tokyo|Tōkyō', t) is not None and re.search(r'\b(la|est|capitale|du)\b', t) is not None, t[:80])),
    dict(id='no_memory', q='Hi! How are you today?', check=lambda t, raw: ('<memory_save' not in raw, 'saved a memory for small talk' if '<memory_save' in raw else '')),
    dict(id='memory', q="My name is Sam and I work as a nurse. Please remember that.", check=lambda t, raw: (re.search(r'<memory_save[^>]*>[^<]*Sam', raw) is not None, '' if re.search(r'<memory_save[^>]*>[^<]*Sam', raw) else 'no memory_save with Sam')),
    dict(id='no_media', q='What kinds of things can you make besides text?', check=lambda t, raw: ('<generate_' not in raw, 'made media unasked' if '<generate_' in raw else '')),
]


# ---- variants: change only the "How to answer" rules (tools and memory sections stay as the page has them) ----
def section(prompt, name):
    a = prompt.index('# ' + name); b = prompt.find('\n# ', a + 1)
    return a, (b if b >= 0 else len(prompt))


def replace_section(prompt, name, new):
    a, b = section(prompt, name)
    return prompt[:a] + new.rstrip() + '\n' + prompt[b:]


VARIANTS = {
    'baseline': lambda p: (p, 0.7),
    'temp03': lambda p: (p, 0.3),
    'short': lambda p: (replace_section(p, 'How to answer', '# How to answer\nBe brief and direct. Follow the requested format exactly.\n'), 0.7),
    'strict': lambda p: (replace_section(p, 'How to answer', '# How to answer\n'
        '- Answer the question first, in as few words as it needs. Short questions get one or two sentences.\n'
        '- When the user asks for a format (one sentence, a list, only JSON, a language), follow it exactly and add nothing around it.\n'
        '- Use Markdown only when it helps: lists for steps, tables for comparisons, code blocks for code.\n'
        "- You run offline on the user's PC: you can't see live data (weather, news, prices) or anything about the user they haven't told you. Say so plainly.\n"
        '- If you are unsure, say so instead of guessing.\n'), 0.7),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--variants', default='baseline,short,strict', help=', '.join(VARIANTS))
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--port', type=int, default=7860)
    ap.add_argument('--model'); ap.add_argument('--ctx', type=int, default=8192)
    a = ap.parse_args()
    if a.model:
        r = post(a.port, '/api/load', {'path': a.model, 'ctx': a.ctx})
        if not r.get('loaded'): raise SystemExit('load failed: ' + json.dumps(r)[:300])
    st = get(a.port, '/api/status')
    if not st.get('loaded'): raise SystemExit('no model loaded in Ithaca (use --model)')
    model = st.get('name') or os.path.basename(st.get('model', ''))
    base = page_prompt()
    vs = [v.strip() for v in a.variants.split(',')]
    for v in vs:
        if v not in VARIANTS: raise SystemExit(f'unknown variant {v}; have: {", ".join(VARIANTS)}')
    outdir = os.path.join(HERE, 'results', 'chat-' + time.strftime('%Y%m%d-%H%M%S')); os.makedirs(outdir, exist_ok=True)
    rows = []
    total = a.runs * len(QUESTIONS) * len(vs); n = 0
    print(f'{model}: {len(QUESTIONS)} questions x {len(vs)} variants x {a.runs} runs = {total} -> {outdir}', flush=True)
    for run in range(a.runs):
        for qd in QUESTIONS:
            for v in vs:
                n += 1
                system, temp = VARIANTS[v](base)
                t0 = time.time()
                raw, stats = chat(a.port, system, qd['q'], temp)
                t = body_of(raw)
                try: ok, note = qd['check'](t, raw)
                except Exception as e: ok, note = False, 'check error ' + repr(e)[:80]
                row = dict(variant=v, q=qd['id'], run=run + 1, passed=bool(ok), note=note, seconds=round(time.time() - t0, 1), words=words(t), answer=t[:800], raw=raw[:1500])
                rows.append(row)
                with open(os.path.join(outdir, 'results.jsonl'), 'a', encoding='utf8') as f: f.write(json.dumps(row) + '\n')
                print(f'[{n}/{total}] {v:9} {qd["id"]:12} {"PASS" if ok else "fail"}  {note[:70]}', flush=True)
    L = [f'# Chat eval: {model}', '', f'{len(QUESTIONS)} questions, {a.runs} runs each.', '', '| Variant | Pass rate | Avg words | Avg seconds |', '|---|---|---|---|']
    for v in vs:
        rs = [r for r in rows if r['variant'] == v]
        L.append(f'| {v} | {sum(r["passed"] for r in rs) / len(rs):.0%} | {sum(r["words"] for r in rs) / len(rs):.0f} | {sum(r["seconds"] for r in rs) / len(rs):.1f} |')
    L += ['', '| Question | ' + ' | '.join(vs) + ' |', '|---|' + '---|' * len(vs)]
    for qd in QUESTIONS:
        L.append(f'| {qd["id"]} | ' + ' | '.join(f'{sum(r["passed"] for r in rows if r["variant"] == v and r["q"] == qd["id"]) / a.runs:.0%}' for v in vs) + ' |')
    L += ['', '## Failures', ''] + [f'- {r["variant"]} / {r["q"]} / run {r["run"]}: {r["note"]}' for r in rows if not r['passed']]
    open(os.path.join(outdir, 'summary.md'), 'w', encoding='utf8').write('\n'.join(L) + '\n')
    print('\n'.join(L[:6 + len(vs)]) + '\n\nfull summary: ' + os.path.join(outdir, 'summary.md'))


if __name__ == '__main__':
    main()
