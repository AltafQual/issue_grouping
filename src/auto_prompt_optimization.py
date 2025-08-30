import dspy

from qgenie import ChatMessage, QGenieClient
from src.constants import QGENEIE_API_KEY

# lm = dspy.LM("custom/Pro", api_key=QGENEIE_API_KEY, api_base="https://qgenie-chat.qualcomm.com/", provider="custom")
# dspy.configure(lm=lm)


class MyLM(dspy.BaseLM):
    def convert_qgenie_to_openai(self, response):
        updated_response = {
            "id": response.id,
            "object": response.object,
            "created": int(response.created),
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }
        if hasattr(response, "choices"):
            updated_response["choices"] = [
                {
                    "index": choice.index,
                    "message": {"role": choice.message.role, "content": choice.message.content},
                    "finish_reason": (
                        str(choice.finish_reason.value) if hasattr(choice.finish_reason, "value") else "stop"
                    ),
                }
                for choice in response.choices
            ]
        else:
            updated_response["choices"] = []
        return updated_response

    def forward(self, prompt, messages=None, **kwargs):
        print(f"Received prompt: {prompt}, messages: {messages}")
        if not isinstance(messages, list):
            messages = [messages]

        # import pdb; pdb.set_trace()
        client = QGenieClient(api_key=QGENEIE_API_KEY)
        response = client.chat(messages=[ChatMessage(role="user", content=message) for message in messages])
        # TODO: return openai kind of response
        return self.convert_qgenie_to_openai(response)


class ClusterNaming(dspy.Signature):
    """Give a name to the Group of similar log Messages"""

    logs: list[str] = dspy.InputField()
    cluster_name: str = dspy.OutputField(desc="Concise name for cluster of log messages in Pascal Case")
