from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Sequence, Union, DefaultDict
from collections import defaultdict
from onnxscript.rewriter import pattern


# --- Priority wrapper ---
@dataclass(frozen=True)
class PRule:
    rule: pattern.RewriteRule
    priority: int = 0  # default priority


RuleLike = Union[pattern.RewriteRule, PRule]

_RULE_PROVIDERS: List[Callable[[], Sequence[RuleLike]]] = []


def register_rules(provider: Callable[[], Sequence[RuleLike]]):
    _RULE_PROVIDERS.append(provider)
    return provider

def collect_rule_buckets(default_priority: int = 0) -> List[pattern.RewriteRuleSet]:
    buckets: DefaultDict[int, List[pattern.RewriteRule]] = defaultdict(list)

    for prov in _RULE_PROVIDERS:
        for item in prov() or []:
            if isinstance(item, PRule):
                p, r = item.priority, item.rule
            else:
                p, r = default_priority, item
            buckets[p].append(r)

    # Highest priority first
    ordered_priorities = sorted(buckets.keys(), reverse=True)
    return [
        pattern.RewriteRuleSet(buckets[p], commute=False) for p in ordered_priorities
    ]
