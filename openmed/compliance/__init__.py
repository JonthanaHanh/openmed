"""Privacy compliance helpers and evidence-generation workflows.

This package hosts local-first, access-logged compliance workflows built on the
surrogate vault and a tamper-evident audit chain, plus deterministic technical
control crosswalks for auditor handoff.
"""

from .audit_chain import (
    AuditRecord,
    AuditSink,
    HashChainAuditLog,
)
from .dsar import (
    DSAR_ADVISORY,
    ERASURE_PREVIEW_EVENT,
    EXPORT_EVENT,
    DsarEntry,
    DsarPackage,
    ErasurePlan,
    SubjectIdentifier,
    assemble_dsar_package,
    plan_erasure,
    render_dsar_summary,
)
from .iso27701 import (
    CONTROL_STATUSES,
    MANIFEST_FILENAME,
    MARKDOWN_FILENAME,
    ControlEvidence,
    ControlEvidencePack,
    ControlEvidencePackResult,
    EvidencePointer,
    build_control_evidence_pack,
    generate_control_evidence_pack,
    load_control_evidence_schema,
    render_control_evidence_markdown,
)

__all__ = [
    "AuditRecord",
    "AuditSink",
    "HashChainAuditLog",
    "DSAR_ADVISORY",
    "EXPORT_EVENT",
    "ERASURE_PREVIEW_EVENT",
    "SubjectIdentifier",
    "DsarEntry",
    "DsarPackage",
    "ErasurePlan",
    "assemble_dsar_package",
    "render_dsar_summary",
    "plan_erasure",
    "CONTROL_STATUSES",
    "MANIFEST_FILENAME",
    "MARKDOWN_FILENAME",
    "ControlEvidence",
    "ControlEvidencePack",
    "ControlEvidencePackResult",
    "EvidencePointer",
    "build_control_evidence_pack",
    "generate_control_evidence_pack",
    "load_control_evidence_schema",
    "render_control_evidence_markdown",
]
