from my_assistant.agents.lead_agent.prompt import apply_prompt_template
from my_assistant.config.app_config import get_app_config


def test_default_prompt_positions_product_as_local_personal_assistant() -> None:
    prompt = apply_prompt_template(app_config=get_app_config(), available_skills=set())

    assert "You are my_assistant, the AI runtime of my_assistant" in prompt
    assert "a local-first personal assistant" in prompt
    assert "structured resumable tasks" in prompt
    assert "transactional file-write" in prompt
    assert "recovery" in prompt
    assert "open-source super agent" not in prompt
    assert "bytedance/my-assistant" not in prompt
    assert "my-assistant.dev" not in prompt
