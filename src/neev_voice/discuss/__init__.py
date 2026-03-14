"""Discuss subcommand package for interactive research discussion.

Implements a state machine with session management for research
document discussion, TTS presentation, and interactive enquiry.
"""

from neev_voice.discuss.enquiry import EnquiryEngine, EnquiryResult
from neev_voice.discuss.migration import (
    CURRENT_SCHEMA_VERSION,
    migrate_concepts_file,
    migrate_session_data,
)
from neev_voice.discuss.names import generate_session_name
from neev_voice.discuss.portability import export_session, import_session
from neev_voice.discuss.prepare import ConceptInfo, PrepareEngine
from neev_voice.discuss.prepare_enquiry import PrepareEnquiryEngine
from neev_voice.discuss.presentation import PresentationEngine, PresentationResult
from neev_voice.discuss.runner import DiscussRunner
from neev_voice.discuss.session import SessionInfo, SessionManager
from neev_voice.discuss.state import DiscussState, StateSnapshot, StateStack

__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "ConceptInfo",
    "DiscussRunner",
    "DiscussState",
    "EnquiryEngine",
    "EnquiryResult",
    "PrepareEngine",
    "PrepareEnquiryEngine",
    "PresentationEngine",
    "PresentationResult",
    "SessionInfo",
    "SessionManager",
    "StateSnapshot",
    "StateStack",
    "export_session",
    "generate_session_name",
    "import_session",
    "migrate_concepts_file",
    "migrate_session_data",
]
