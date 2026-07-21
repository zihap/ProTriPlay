from api_client import get_client, handle_stream_response


class BaseAgent:
    """
    基础智能体类，所有智能体的基类
    
    提供智能体的基本功能：
    - 对话历史管理：记录和查询对话记录
    - API客户端交互：通过generate_response方法与LLM进行交互
    - 统一的智能体标识：通过name属性区分不同智能体实例
    """
    def __init__(self, name: str):
        self.name = name
        self.client = get_client()
        self.dialogue_history = []

    def add_dialogue_record(self, speaker: str, record_type: str, content: str, target: str = None):
        record = {
            "time": len(self.dialogue_history),
            "speaker": speaker,
            "record_type": record_type,
            "content": content,
        }
        if target and (record_type == "对话" or record_type.startswith("对")):
            record["target"] = target
        self.dialogue_history.append(record)

    def get_dialogue_history(self, limit: int = 10):
        return self.dialogue_history[-limit:] if self.dialogue_history else []

    def get_all_dialogue_history(self):
        return self.dialogue_history if self.dialogue_history else []

    def generate_response(self, system_prompt: str, user_prompt: str, extra_body=None):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return handle_stream_response(self.client, "", messages, extra_body)

    def __str__(self):
        return f"{self.__class__.__name__}(name={self.name})"
