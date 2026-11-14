"""System-instruction variants to compare. Each takes the Coding page's own prompt and tools (base = {'prompt',
'tools'}) and the workspace name, and returns {'system': str, 'temperature': float}. Add one here and pass its
name with --variants."""
import json


def tools_json(base):
    return '\n'.join(json.dumps({'type': 'function', 'function': {'name': n, 'description': d, 'parameters': {
        'type': 'object', 'properties': {k: {'type': t, 'description': dd} for k, (t, dd) in p.items()}, 'required': r}}}) for n, d, p, r in base['tools'])


JSON_CALLS = '''# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>'''

# Qwen3-Coder's own tool-call style (its chat template uses this)
XML_CALLS = '''# Tools

You have access to the following functions:

<tools>
{tools}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>'''

RULES = base_rules = None


def rules_of(base):
    """The page prompt without its tool section: the working rules only."""
    return base['prompt'].split('# Tools')[0].rstrip()


def baseline(base, name):
    return {'system': base['prompt'], 'temperature': 0.2}


def temp0(base, name):
    return {'system': base['prompt'], 'temperature': 0.0}


def short(base, name):
    return {'system': f'You are a coding agent working in the folder "{name}". Use the tools to look at files and change them. '
                      f'When the task is done, reply with a short summary and no tool call.\n\n' + JSON_CALLS.format(tools=tools_json(base)),
            'temperature': 0.2}


def xml(base, name):
    return {'system': rules_of(base) + '\n\n' + XML_CALLS.format(tools=tools_json(base)), 'temperature': 0.2}


EXAMPLE = '''
Example of a good session (a different project):
User: The greeting is printed twice; fix it.
You: Let me look at the code. <tool_call>{"name": "search", "arguments": {"pattern": "print("}}</tool_call>
Tool: app.py:3:     print(greet())
      app.py:4:     print(greet())
You: Line 4 repeats line 3. <tool_call>{"name": "edit_file", "arguments": {"path": "app.py", "old": "    print(greet())\\n    print(greet())", "new": "    print(greet())"}}</tool_call>
Tool: edited app.py
You: <tool_call>{"name": "run_command", "arguments": {"command": "python app.py"}}</tool_call>
Tool: Hello  [exit code 0]
You: Fixed: app.py printed the greeting twice (line 4 duplicated line 3); I removed the extra line and checked that it now prints once.'''


def example(base, name):
    return {'system': rules_of(base) + '\n' + EXAMPLE + '\n\n' + JSON_CALLS.format(tools=tools_json(base)), 'temperature': 0.2}


def plan(base, name):
    extra = ('\n- Before the first tool call, write a short plan (2-4 numbered steps).'
             '\n- Before your final summary, re-read or test what you changed to confirm it is right.')
    return {'system': rules_of(base) + extra + '\n\n' + JSON_CALLS.format(tools=tools_json(base)), 'temperature': 0.2}


VARIANTS = {'baseline': baseline, 'temp0': temp0, 'short': short, 'xml': xml, 'example': example, 'plan': plan}
