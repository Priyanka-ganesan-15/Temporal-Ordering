"""Custom exceptions for temporal-ordering workflows."""


class TemporalOrderingError(Exception):
    """Base exception for project errors."""


class ManifestValidationError(TemporalOrderingError):
    """Raised when manifest or frame paths fail validation."""


class SequenceNotFoundError(TemporalOrderingError):
    """Raised when a sequence_id is not present in the manifest."""
