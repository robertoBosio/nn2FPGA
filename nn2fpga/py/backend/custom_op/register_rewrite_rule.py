from typing import Callable, List
from onnxscript.rewriter import pattern

_RULE_PROVIDERS: List[Callable[[], List[pattern.RewriteRule]]] = []

def register_rules(provider: Callable[[], List[pattern.RewriteRule]]):
    _RULE_PROVIDERS.append(provider)
    return provider

def collect_rules() -> pattern.RewriteRuleSet:
    rules: List[pattern.RewriteRule] = []
    for prov in _RULE_PROVIDERS:
        rules.extend(prov() or [])
    return pattern.RewriteRuleSet(rules, commute=False)
