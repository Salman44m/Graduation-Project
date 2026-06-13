import json
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# We will test two scenarios:
# 1. Output from to_json()
d1 = HumanMessage(content="hello").to_json()

# 2. Output from model_dump() or dict() 
# Langchain's message objects actually dump to dicts that look like:
# {"content": "hello", "type": "human", "additional_kwargs": {}}
d2 = HumanMessage(content="hello").dict()

print("d1:", d1)
print("d2:", d2)

# LangChain has messages_from_dict
from langchain_core.messages.utils import messages_from_dict, message_to_dict

print("message_to_dict:", message_to_dict(HumanMessage(content="hi")))
