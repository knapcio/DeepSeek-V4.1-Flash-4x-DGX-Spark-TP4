#!/usr/bin/env python3
"""Required/named tool choice against the real SGLang/xgrammar stack."""
import json

from xgrammar import Grammar, StructuralTag, StructuralTagItem
from xgrammar.testing import _is_grammar_accept_string

from sglang.srt.entrypoints.openai.protocol import (
    Function,
    Tool,
    ToolChoice,
    ToolChoiceFuncName,
)
from sglang.srt.function_call.function_call_parser import FunctionCallParser

SCHEMA = {
    'type': 'object',
    'properties': {'symbol': {'type': 'string'}},
    'required': ['symbol'],
}
TOOLS = [
    Tool(
        type='function',
        function=Function(
            name='lookup_symbol',
            description='Look up a symbol',
            parameters=SCHEMA,
        ),
    ),
    Tool(
        type='function',
        function=Function(
            name='list_files',
            description='List files',
            parameters={
                'type': 'object',
                'properties': {'path': {'type': 'string'}},
            },
        ),
    ),
]
NAMED = ToolChoice(function=ToolChoiceFuncName(name='lookup_symbol'))
failures = []


def check(label, ok):
    print(('ok   ' if ok else 'FAIL ') + label)
    if not ok:
        failures.append(label)


def grammar_for(parser, choice, thinking=False, parallel=True):
    constraint = parser.get_structure_constraint(
        choice,
        parallel_tool_calls=parallel,
        thinking_mode=thinking,
    )
    check(
        f'{choice!r} thinking={thinking} parallel={parallel}: constraint',
        constraint is not None and constraint[0] == 'structural_tag',
    )
    tag = constraint[1]
    check(f'{choice!r}: V4.1 native tag', isinstance(tag, StructuralTag))
    if isinstance(tag, StructuralTag):
        return Grammar.from_structural_tag(tag)
    legacy = json.loads(tag.model_dump_json(by_alias=True))
    items = [
        StructuralTagItem(
            begin=item['begin'],
            schema=json.dumps(item.get('schema') or {}),
            end=item['end'],
        )
        for item in legacy['structures']
    ]
    converted = StructuralTag.from_legacy_structural_tag(
        items, legacy['triggers'])
    converted.format.at_least_one = legacy.get('at_least_one', False)
    return Grammar.from_structural_tag(converted)


def main():
    parser = FunctionCallParser(TOOLS, 'deepseekv41')
    detector = parser.detector

    def invoke(name, arguments):
        return (
            f'{detector.invoke_start_token} name="{name}">'
            f'{json.dumps(arguments)}{detector.invoke_end_token}\n'
        )

    prefix = getattr(detector, 'tool_calls_prefix', '\n\n')

    def block(*calls):
        return f"{prefix}{detector.bot_token}\n{''.join(calls)}{detector.eot_token}"

    one = invoke('lookup_symbol', {'symbol': 'WidgetFactory'})
    bare = (
        f'{detector.invoke_start_token} name="lookup_symbol">'
        f'{{"symbol": "WidgetFactory"}}{detector.invoke_end_token}\n'
        f'{detector.eot_token}'
    )

    for choice in (NAMED, 'required'):
        for thinking in (False, True):
            grammar = grammar_for(parser, choice, thinking=thinking)
            reasoning = 'Need the symbol.</think>' if thinking else ''
            check(
                f'{choice!r} thinking={thinking}: accepts calls block',
                _is_grammar_accept_string(grammar, reasoning + block(one)),
            )
            check(
                f'{choice!r} thinking={thinking}: rejects bare invoke',
                not _is_grammar_accept_string(grammar, reasoning + bare),
            )
            check(
                f'{choice!r} thinking={thinking}: rejects prose',
                not _is_grammar_accept_string(
                    grammar, reasoning + 'I will look it up.'),
            )

    grammar = grammar_for(parser, NAMED)
    check(
        'named: rejects a different tool',
        not _is_grammar_accept_string(
            grammar, block(invoke('list_files', {'path': '.'}))),
    )
    two = block(one, invoke('list_files', {'path': 'src'}))
    grammar = grammar_for(parser, 'required', parallel=True)
    check('required parallel: accepts two invokes',
          _is_grammar_accept_string(grammar, two))
    grammar = grammar_for(parser, 'required', parallel=False)
    check('required non-parallel: rejects two invokes',
          not _is_grammar_accept_string(grammar, two))

    text, calls = parser.parse_non_stream(block(one))
    check(
        'parse: one call, no leftover markup',
        len(calls) == 1
        and calls[0].name == 'lookup_symbol'
        and json.loads(calls[0].parameters) == {'symbol': 'WidgetFactory'}
        and detector.invoke_start_token not in text,
    )
    _, calls = parser.parse_non_stream(two)
    check('parse: two parallel calls',
          [call.name for call in calls] == ['lookup_symbol', 'list_files'])

    if failures:
        raise SystemExit(f'{len(failures)} failure(s)')
    print('PASS test_v41_tool_constraint')


if __name__ == '__main__':
    main()
