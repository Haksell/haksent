import os

from anthropic import Anthropic
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

SYSTEM_MESSAGE = {"role": "system", "content": "You are a helpful assistant."}
FIRST_MESSAGE = {
    "role": "user",
    "content": "You will be connected to Claude Haiku 5.5 in a loop. "
    "Talk to him as if you were a user. "
    "Brainstorm videogame ideas with him, without letting him know you are an AI too. "
    "Your first question starts now.",
}


def main():
    load_dotenv(find_dotenv())
    openai_client = OpenAI()
    anthropic_client = Anthropic()
    width = os.get_terminal_size().columns
    separator = "=" * width

    chat_messages = []
    for _ in range(5):
        openai_response = openai_client.chat.completions.create(
            model="gpt-6-luna",
            max_completion_tokens=2048,
            messages=[
                SYSTEM_MESSAGE,  # type: ignore
                FIRST_MESSAGE,
                *[
                    {"role": "user" if i & 1 else "assistant", "content": m}
                    for i, m in enumerate(chat_messages)
                ],
            ],
        )
        text = openai_response.choices[0].message.content
        assert text is not None
        chat_messages.append(text)
        print(text)
        print(separator)

        anthropic_response = anthropic_client.messages.create(
            model="claude-haiku-5-5",
            max_tokens=2048,
            messages=[
                {"role": "assistant" if i & 1 else "user", "content": m}
                for i, m in enumerate(chat_messages)
            ],  # type: ignore
        )
        text = "".join(
            block.text for block in anthropic_response.content if block.type == "text"
        )
        chat_messages.append(text)
        print(text)
        print(separator)


if __name__ == "__main__":
    main()
