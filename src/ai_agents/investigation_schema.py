"""Pydantic models for one data-quality investigation report.

Confirmed measurements live in observed_evidence. They are copied from the
validated finding after the model responds, so the model cannot replace the
rule id, counts, metric, threshold, or evidence.

The narrative is the only part the model writes. Root-cause items are
hypotheses, and each one must include a verification step.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from .schemas import Finding

NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
ConfidenceLevel = Literal["low", "medium", "high"]


class RootCauseHypothesis(BaseModel):
    """One possible cause, plus how someone could check it.

    A hypothesis is not a confirmed fact. The verification step says what to
    look at next. It does not prove the hypothesis.
    """

    model_config = ConfigDict(extra="forbid")

    hypothesis: NonBlank = Field(
        description="A possible cause. This is not a confirmed fact."
    )
    verification_step: NonBlank = Field(
        description="A concrete check that would support or reject this hypothesis."
    )


class InvestigationNarrative(BaseModel):
    """Text the language model is allowed to write.

    Rule ids, counts, rates, and thresholds are intentionally absent. Those
    values are copied from the validated finding.
    """

    model_config = ConfigDict(extra="forbid")

    issue_summary: NonBlank = Field(
        description="Plain-language explanation of the observed data-quality problem."
    )
    business_impact: NonBlank = Field(
        description="Possible effect on reporting or decisions. This is an implication, not a measured loss."
    )
    possible_root_causes: list[RootCauseHypothesis] = Field(min_length=1)
    recommended_investigation_steps: list[NonBlank] = Field(min_length=1)
    recommended_remediation: list[NonBlank] = Field(min_length=1)
    confidence_level: ConfidenceLevel
    limitations: NonBlank = Field(
        description=(
            "What is uncertain. Include that root causes are hypotheses and "
            "that the data is synthetic."
        )
    )


class InvestigationReport(BaseModel):
    """Report saved for one validated finding."""

    model_config = ConfigDict(extra="forbid")

    rule_id: NonBlank
    issue_summary: NonBlank
    observed_evidence: Finding = Field(
        description=(
            "Exact copy of the validated finding, including metrics and evidence. "
            "Not rewritten by the model."
        )
    )
    business_impact: NonBlank
    possible_root_causes: list[RootCauseHypothesis] = Field(min_length=1)
    recommended_investigation_steps: list[NonBlank] = Field(min_length=1)
    recommended_remediation: list[NonBlank] = Field(min_length=1)
    confidence_level: ConfidenceLevel
    limitations: NonBlank

    @model_validator(mode="after")
    def rule_id_matches_evidence(self) -> InvestigationReport:
        if self.rule_id != self.observed_evidence.rule_id:
            raise ValueError(
                f"rule_id {self.rule_id!r} does not match observed evidence "
                f"rule_id {self.observed_evidence.rule_id!r}."
            )
        return self
