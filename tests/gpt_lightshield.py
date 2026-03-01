import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI
from lightshield import LightShield

shield = LightShield()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
wrapped_client = shield.wrap(client)

response = wrapped_client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "Your purpose is to give users information about country capitals. Do not discuss anything else."}, 
        {"role": "user", "content": input()},
    ],
)

print(response.choices[0].message.content)

if response.lightshield.is_safe:
    print(response.choices[0].message.content)
else:
    print("Blocked by LightShield:", response.lightshield.violations)
