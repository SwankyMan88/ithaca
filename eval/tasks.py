"""Coding-agent tasks: each builds a small project in an empty folder, gives the agent one request, and checks the
result with code the agent never sees. check(folder, reply) returns (passed, note)."""
import json, os, re, subprocess, sys, zipfile


def w(root, rel, text):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w', encoding='utf8', newline='') as f:
        f.write(text)


def r(root, rel):
    with open(os.path.join(root, rel), encoding='utf8', newline='') as f:
        return f.read()


def py(root, code):
    """Runs a check snippet with the project folder as the working directory and on sys.path."""
    p = subprocess.run([sys.executable, '-I', '-c', 'import sys; sys.path.insert(0, "."); ' + code], cwd=root, capture_output=True, text=True, timeout=60)
    return p.returncode == 0, (p.stdout + p.stderr).strip()[-300:]


# ---- Office fixtures (built by hand: no libraries needed) ----
CT_DOCX = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>'
RELS_DOCX = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'


def docx(path, paras):
    body = ''.join(f'<w:p><w:r>{"<w:rPr><w:b/></w:rPr>" if i == 0 else ""}<w:t xml:space="preserve">{t}</w:t></w:r></w:p>' for i, t in enumerate(paras))
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', CT_DOCX); z.writestr('_rels/.rels', RELS_DOCX)
        z.writestr('word/document.xml', f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>{body}</w:body></w:document>')


def docx_paras(path):
    x = zipfile.ZipFile(path).read('word/document.xml').decode()
    return [re.sub(r'<[^>]+>', '', p) for p in re.findall(r'<w:p>(.*?)</w:p>', x)], x


def xlsx(path, rows):
    def cell(ref, v):
        return f'<c r="{ref}"><v>{v}</v></c>' if isinstance(v, (int, float)) else f'<c r="{ref}" t="inlineStr"><is><t>{v}</t></is></c>'
    data = ''.join(f'<row r="{i + 1}">' + ''.join(cell(chr(65 + j) + str(i + 1), v) for j, v in enumerate(row)) + '</row>' for i, row in enumerate(rows))
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>')
        z.writestr('_rels/.rels', '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        z.writestr('xl/workbook.xml', '<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Budget" sheetId="1" r:id="rId1"/></sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels', '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>')
        z.writestr('xl/worksheets/sheet1.xml', f'<?xml version="1.0" encoding="UTF-8"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>{data}</sheetData></worksheet>')


def xlsx_cells(path):
    x = zipfile.ZipFile(path).read('xl/worksheets/sheet1.xml').decode()
    out = {}
    for ref, body in re.findall(r'<c r="([A-Z]+\d+)"[^>]*>(.*?)</c>', x):
        v = re.search(r'<v>(.*?)</v>', body); t = re.search(r'<t[^>]*>(.*?)</t>', body); f = re.search(r'<f>(.*?)</f>', body)
        out[ref] = '=' + f.group(1) if f else (t.group(1) if t else (v.group(1) if v else ''))
    return out


# ---- the projects ----
CART = '''class Cart:
    def __init__(self):
        self.items = {}

    def add(self, name, price, qty=1):
        if name in self.items:
            self.items[name]["qty"] = qty
        else:
            self.items[name] = {"price": price, "qty": qty}

    def total(self):
        return sum(i["price"] for i in self.items.values())

    def remove(self, name):
        del self.items[name]
'''
CART_TEST = '''import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shop.cart import Cart

def test_total():
    c = Cart()
    c.add("apple", 0.5, 4)
    c.add("pear", 1.25)
    c.add("apple", 0.5, 2)
    assert abs(c.total() - 4.25) < 1e-9, c.total()

if __name__ == "__main__":
    test_total(); print("ok")
'''


def cart_project(root):
    w(root, 'shop/__init__.py', '')
    w(root, 'shop/cart.py', CART)
    w(root, 'tests/test_cart.py', CART_TEST)
    w(root, 'README.md', '# Shop\n\nA tiny shopping cart. Run the tests with `python tests/test_cart.py`.\n')


def fixed_cart_project(root):
    """The cart without its bugs: for tasks that build on it."""
    cart_project(root)
    w(root, 'shop/cart.py', CART.replace('self.items[name]["qty"] = qty', 'self.items[name]["qty"] += qty').replace('i["price"] for i', 'i["price"] * i["qty"] for i'))
    w(root, 'tests/test_cart.py', CART_TEST)


def chk_fix_bugs(root, reply):
    if r(root, 'tests/test_cart.py') != CART_TEST:
        return False, 'changed the test instead of the code'
    return py(root, 'from shop.cart import Cart; c=Cart(); c.add("a",2,3); c.add("a",2,1); c.add("b",1); assert c.total()==9, c.total(); c.remove("b"); assert c.total()==8')


def chk_discount(root, reply):
    ok, note = py(root, 'from shop.cart import Cart; c=Cart(); c.add("a",10,2); c.apply_discount(25); assert abs(c.total()-15)<1e-9, c.total(); c.apply_discount(0); assert abs(c.total()-20)<1e-9, c.total()')
    if not ok:
        return False, note
    return py(root, 'from shop.cart import Cart\ntry:\n    Cart().apply_discount(150)\nexcept ValueError:\n    pass\nelse:\n    raise SystemExit("no ValueError for 150")')


def tax_project(root):
    w(root, 'billing/tax.py', 'RATE = 0.2\n\n\ndef calc_tax(amount):\n    """Tax owed on an amount."""\n    return round(amount * RATE, 2)\n')
    w(root, 'billing/invoice.py', 'from billing.tax import calc_tax\n\n\ndef invoice_total(net):\n    return net + calc_tax(net)\n')
    w(root, 'billing/__init__.py', '')
    w(root, 'report.py', 'from billing.invoice import invoice_total\nfrom billing.tax import calc_tax\n\nprint("tax on 100:", calc_tax(100))\nprint("total:", invoice_total(100))\n')


def chk_rename(root, reply):
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.endswith('.py') and 'calc_tax' in r(root, os.path.relpath(os.path.join(dp, f), root)):
                return False, 'calc_tax still in ' + f
    ok, note = py(root, 'from billing.tax import compute_tax; from billing.invoice import invoice_total; assert compute_tax(100)==20; assert invoice_total(100)==120')
    if not ok:
        return False, note
    p = subprocess.run([sys.executable, '-I', 'report.py'], cwd=root, capture_output=True, text=True, timeout=60)
    return (p.returncode == 0 and '120' in p.stdout), p.stdout[-120:] + p.stderr[-200:]


def docs_project(root):
    docx(os.path.join(root, 'report.docx'), ['Quarterly Report', 'Sales grew by 12 percent and costs fell.', 'Next steps are listed below.'])


def chk_docx(root, reply):
    paras, x = docx_paras(os.path.join(root, 'report.docx'))
    if paras[:1] != ['2026 Annual Report']:
        return False, f'title is {paras[:1]}'
    if paras[1:] != ['Sales grew by 12 percent and costs fell.', 'Next steps are listed below.']:
        return False, 'other paragraphs changed'
    return ('<w:b/>' in x), 'bold kept' if '<w:b/>' in x else 'title lost its bold'


def budget_project(root):
    xlsx(os.path.join(root, 'budget.xlsx'), [['Item', 'Cost'], ['Rent', 1200], ['Food', 450]])


def chk_xlsx(root, reply):
    c = xlsx_cells(os.path.join(root, 'budget.xlsx'))
    rows = {}
    for ref, v in c.items():
        rows.setdefault(int(re.sub(r'[A-Z]', '', ref)), {})[re.sub(r'\d', '', ref)] = v
    found = [n for n, row in rows.items() if row.get('A', '').strip().lower() == 'utilities' and str(row.get('B', '')).strip() in ('180', '180.0')]
    if not found:
        return False, f'no Utilities 180 row: {rows}'
    if c.get('A2') != 'Rent' or c.get('B2') != '1200' or c.get('A3') != 'Food':
        return False, 'existing rows changed'
    return True, f'row {found[0]}'


def config_project(root):
    w(root, 'config.json', '{\n  "name": "api",\n  "server": {\n    "host": "127.0.0.1",\n    "port": 8080\n  },\n  "debug": false\n}\n')
    w(root, 'app.js', 'const cfg = require("./config.json");\nconsole.log(`listening on ${cfg.server.host}:${cfg.server.port}`);\n')


def chk_config(root, reply):
    try:
        j = json.loads(r(root, 'config.json'))
    except Exception as e:
        return False, 'config.json is no longer valid JSON: ' + str(e)
    if j.get('server', {}).get('port') != 9090:
        return False, f'port is {j.get("server", {}).get("port")!r}'
    if j.get('name') != 'api' or j.get('debug') is not False or j['server'].get('host') != '127.0.0.1':
        return False, 'other settings changed'
    return True, ''


def chk_readme(root, reply):
    p = os.path.join(root, 'README.md')
    if not os.path.exists(p):
        return False, 'no README.md'
    t = r(root, 'README.md').lower()
    miss = [k for k in ('usage', 'calc_tax', 'report.py') if k not in t]
    return not miss, ('missing: ' + ', '.join(miss)) if miss else ''


def chk_question(root, reply, before):
    if snapshot(root) != before:
        return False, 'changed files for a question'
    t = reply.lower()
    return ('qty' in t or 'quantity' in t), 'answer does not mention quantity' if not ('qty' in t or 'quantity' in t) else ''


def snapshot(root):
    out = {}
    for dp, dn, fs in os.walk(root):
        dn[:] = [d for d in dn if d != '__pycache__']
        for f in fs:
            p = os.path.join(dp, f)
            with open(p, 'rb') as h:
                out[os.path.relpath(p, root)] = h.read()
    return out


TASKS = [
    dict(id='fix_bugs', setup=cart_project, prompt='The test in tests/test_cart.py fails. Find the bugs in the cart and fix them.', check=chk_fix_bugs),
    dict(id='feature', setup=fixed_cart_project, prompt='Add an apply_discount(percent) method to Cart: it applies a percentage discount to total(). 0 means no discount, and anything outside 0-100 must raise ValueError.', check=chk_discount),
    dict(id='rename', setup=tax_project, prompt='Rename the function calc_tax to compute_tax everywhere in the project.', check=chk_rename),
    dict(id='docx', setup=docs_project, prompt='In report.docx, change the title to "2026 Annual Report".', check=chk_docx),
    dict(id='xlsx', setup=budget_project, prompt='In budget.xlsx, add a row for Utilities costing 180.', check=chk_xlsx),
    dict(id='config', setup=config_project, prompt='Change the server port in config.json to 9090.', check=chk_config),
    dict(id='readme', setup=tax_project, prompt='Write a README.md for this project: a short description and a Usage section that shows how to run report.py and how to call calc_tax from Python.', check=chk_readme),
    dict(id='question', setup=fixed_cart_project, prompt='How does Cart.total() work? Just explain, do not change any files.', check=None),   # check: chk_question (no file may change)
]

