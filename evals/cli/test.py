from evals.predictors.multimodal.llama3v_agent import Llama3VAgent
from evals.predictors.multimodal.llama3_agent import Llama3Agent

from autogen import AssistantAgent, UserProxyAgent
from autogen import GroupChatManager, GroupChat

if __name__ == "__main__":
    config = [
        {
            "model": "/checkpoint/maestro/models/Meta-Llama-3.1-8B-Instruct-hf",
            "model_client_cls": "Llama3Client",
        },
        {
            "model": "/checkpoint/maestro/models/Meta-Llama-3.2-11B-Vision-Instruct",
            "model_client_cls": "Llama3VClient",
        },
    ]

    bob = Llama3Agent(
        "bob",
        system_message="A chatbot named bob",
        llm_config={"config_list": config},
    )

    user_proxy = UserProxyAgent(
        "user_proxy",
        code_execution_config=False,  # no code execution
    )

    user_proxy.initiate_chat(bob, message="hi, how's the weather today?")
