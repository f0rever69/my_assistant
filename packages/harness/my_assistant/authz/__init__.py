"""Pluggable fine-grained authorization (resource-level RBAC and beyond)."""

from my_assistant.authz.adapter import GuardrailAuthorizationAdapter
from my_assistant.authz.enforcement import filter_tools_by_authorization
from my_assistant.authz.principal import build_principal_from_context, normalize_authz_attributes
from my_assistant.authz.provider import AuthorizationProvider, AuthzDecision, AuthzReason, AuthzRequest, Principal
from my_assistant.authz.rbac import RbacAuthorizationProvider
from my_assistant.authz.runtime import resolve_authorization_provider
from my_assistant.authz.sandbox_authz import authorize_sandbox_execution
from my_assistant.authz.tool_filter import apply_tool_authorization

__all__ = [
    "AuthzDecision",
    "AuthzReason",
    "AuthzRequest",
    "AuthorizationProvider",
    "GuardrailAuthorizationAdapter",
    "Principal",
    "RbacAuthorizationProvider",
    "apply_tool_authorization",
    "authorize_sandbox_execution",
    "build_principal_from_context",
    "filter_tools_by_authorization",
    "normalize_authz_attributes",
    "resolve_authorization_provider",
]
