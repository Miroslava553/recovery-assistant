from __future__ import annotations

import unittest

from app_core.evidence import (
    LEVEL_RULES,
    EvidenceSource,
    EvidenceSummary,
    RecoveryEvidence,
    WorkloadLevel,
    derive_level,
    derive_level_from_evidence,
)


def summary(
    *,
    codes: set[str] | None = None,
    sources: set[EvidenceSource] | None = None,
    max_severity: int = 0,
    typing_count: int = 0,
    has_any: bool | None = None,
) -> EvidenceSummary:
    """Собрать свёртку вручную, без построения полноценных сигналов."""

    resolved_codes = frozenset(codes or set())
    resolved_sources = frozenset(sources or set())
    if has_any is None:
        has_any = bool(resolved_codes or resolved_sources)
    return EvidenceSummary(
        codes=resolved_codes,
        sources=resolved_sources,
        max_severity=max_severity,
        typing_count=typing_count,
        has_any=has_any,
    )


def evidence(
    code: str,
    source: EvidenceSource,
    severity: int = 1,
    *,
    decision_ready: bool = True,
) -> RecoveryEvidence:
    return RecoveryEvidence(
        code=code,
        source=source,
        severity=severity,
        title=code,
        detail=code,
        decision_ready=decision_ready,
    )


class LevelRuleTableTests(unittest.TestCase):
    """По одному тесту на каждое правило таблицы."""

    def test_very_high_self_report_is_priority_and_urgent(self) -> None:
        decision = derive_level(summary(codes={"self_report_very_high"}))
        self.assertEqual(decision.level, WorkloadLevel.RECOVERY_PRIORITY)
        self.assertEqual(decision.rule_id, "very_high_self_report")
        self.assertTrue(decision.urgent)

    def test_severe_eye_closure_is_priority_and_urgent(self) -> None:
        decision = derive_level(summary(codes={"severe_eye_closure"}))
        self.assertEqual(decision.level, WorkloadLevel.RECOVERY_PRIORITY)
        self.assertEqual(decision.rule_id, "severe_eye_closure")
        self.assertTrue(decision.urgent)

    def test_strong_self_report_is_expressed(self) -> None:
        decision = derive_level(summary(codes={"self_report_strong"}))
        self.assertEqual(decision.level, WorkloadLevel.EXPRESSED)
        self.assertEqual(decision.rule_id, "strong_self_report")
        self.assertFalse(decision.urgent)

    def test_two_independent_channels_are_expressed(self) -> None:
        decision = derive_level(
            summary(
                codes={"long_continuous_work", "typing_slowdown"},
                sources={EvidenceSource.SESSION, EvidenceSource.TYPING},
                max_severity=2,
            )
        )
        self.assertEqual(decision.level, WorkloadLevel.EXPRESSED)
        self.assertEqual(decision.rule_id, "two_independent_channels")

    def test_two_channels_need_moderate_severity(self) -> None:
        """Два канала со слабыми признаками не дают четвёртый уровень."""

        decision = derive_level(
            summary(
                codes={"typing_slowdown"},
                sources={EvidenceSource.SESSION, EvidenceSource.TYPING},
                max_severity=1,
            )
        )
        self.assertNotEqual(decision.level, WorkloadLevel.EXPRESSED)

    def test_very_long_session_is_expressed(self) -> None:
        decision = derive_level(summary(codes={"very_long_continuous_work"}))
        self.assertEqual(decision.level, WorkloadLevel.EXPRESSED)
        self.assertEqual(decision.rule_id, "very_long_continuous_work")

    def test_long_session_is_sustained(self) -> None:
        decision = derive_level(summary(codes={"long_continuous_work"}))
        self.assertEqual(decision.level, WorkloadLevel.SUSTAINED)
        self.assertEqual(decision.rule_id, "long_continuous_work")

    def test_two_typing_anomalies_are_sustained(self) -> None:
        decision = derive_level(
            summary(
                codes={"typing_slowdown", "typing_rhythm"},
                sources={EvidenceSource.TYPING},
                typing_count=2,
            )
        )
        self.assertEqual(decision.level, WorkloadLevel.SUSTAINED)
        self.assertEqual(decision.rule_id, "persistent_typing_anomalies")

    def test_single_typing_anomaly_is_only_early(self) -> None:
        decision = derive_level(
            summary(
                codes={"typing_slowdown"},
                sources={EvidenceSource.TYPING},
                typing_count=1,
            )
        )
        self.assertEqual(decision.level, WorkloadLevel.EARLY)

    def test_moderate_self_report_needs_a_second_channel(self) -> None:
        alone = derive_level(
            summary(
                codes={"self_report_moderate"},
                sources={EvidenceSource.SELF_REPORT},
            )
        )
        self.assertEqual(alone.level, WorkloadLevel.EARLY)

        supported = derive_level(
            summary(
                codes={"self_report_moderate"},
                sources={EvidenceSource.SELF_REPORT, EvidenceSource.TYPING},
            )
        )
        self.assertEqual(supported.level, WorkloadLevel.SUSTAINED)
        self.assertEqual(supported.rule_id, "moderate_self_report_with_support")

    def test_any_ready_evidence_is_early(self) -> None:
        decision = derive_level(
            summary(codes={"unclassified"}, sources={EvidenceSource.SESSION})
        )
        self.assertEqual(decision.level, WorkloadLevel.EARLY)
        self.assertEqual(decision.rule_id, "any_ready_evidence")

    def test_no_evidence_is_stable(self) -> None:
        decision = derive_level(summary())
        self.assertEqual(decision.level, WorkloadLevel.STABLE)
        self.assertEqual(decision.rule_id, "no_evidence")
        self.assertFalse(decision.urgent)


class RuleTableIntegrityTests(unittest.TestCase):
    def test_rule_ids_are_unique(self) -> None:
        ids = [rule.rule_id for rule in LEVEL_RULES]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_rule_has_an_explanation(self) -> None:
        for rule in LEVEL_RULES:
            with self.subTest(rule=rule.rule_id):
                self.assertTrue(rule.explanation.strip())

    def test_levels_do_not_increase_further_down_the_table(self) -> None:
        """Порядок строк и есть приоритет: сильные правила стоят выше."""

        levels = [rule.level for rule in LEVEL_RULES]
        self.assertEqual(levels, sorted(levels, reverse=True))

    def test_only_priority_rules_are_urgent(self) -> None:
        for rule in LEVEL_RULES:
            if rule.urgent:
                with self.subTest(rule=rule.rule_id):
                    self.assertEqual(rule.level, WorkloadLevel.RECOVERY_PRIORITY)

    def test_stronger_rule_wins_over_weaker_one(self) -> None:
        decision = derive_level(
            summary(
                codes={"self_report_very_high", "long_continuous_work"},
                sources={EvidenceSource.SELF_REPORT, EvidenceSource.SESSION},
                max_severity=3,
            )
        )
        self.assertEqual(decision.rule_id, "very_high_self_report")


class EvidenceSummaryTests(unittest.TestCase):
    def test_not_ready_evidence_is_ignored(self) -> None:
        items = [
            evidence("long_continuous_work", EvidenceSource.SESSION, decision_ready=False),
        ]
        self.assertEqual(
            derive_level_from_evidence(items).level, WorkloadLevel.STABLE
        )

    def test_summary_counts_typing_sources_only(self) -> None:
        items = [
            evidence("typing_slowdown", EvidenceSource.TYPING),
            evidence("typing_rhythm", EvidenceSource.TYPING),
            evidence("long_continuous_work", EvidenceSource.SESSION),
        ]
        built = EvidenceSummary.from_evidence(items)
        self.assertEqual(built.typing_count, 2)
        self.assertEqual(len(built.sources), 2)
        self.assertTrue(built.has_any)

    def test_summary_takes_maximum_severity(self) -> None:
        items = [
            evidence("a", EvidenceSource.SESSION, severity=1),
            evidence("b", EvidenceSource.TYPING, severity=3),
        ]
        self.assertEqual(EvidenceSummary.from_evidence(items).max_severity, 3)


if __name__ == "__main__":
    unittest.main()
