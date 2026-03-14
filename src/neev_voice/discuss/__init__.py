"""Discuss subcommand package for interactive research discussion.

Implements a state machine with session management for research
document discussion, TTS presentation, and interactive enquiry.
"""

from neev_voice.discuss.enquiry import EnquiryEngine, EnquiryResult
from neev_voice.discuss.names import generate_session_name
from neev_voice.discuss.prepare import ConceptInfo, PrepareEngine
from neev_voice.discuss.prepare_enquiry import PrepareEnquiryEngine
from neev_voice.discuss.presentation import PresentationEngine, PresentationResult
from neev_voice.discuss.runner import DiscussRunner
from neev_voice.discuss.session import SessionInfo, SessionManager
from neev_voice.discuss.state import DiscussState, StateSnapshot, StateStack

__all__ = [
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
    "generate_session_name",
]
