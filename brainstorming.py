import os

from dotenv import find_dotenv, load_dotenv
from litellm import completion
from litellm.types.utils import ModelResponse

SYSTEM_MESSAGE = {"role": "system", "content": "You are a helpful assistant."}
USER_MESSAGE = {
    "role": "user",
    "content": """\
You will be connected to Claude Haiku 5.5 in a loop.
Talk to him as if you were a user.
Brainstorm videogame ideas with him, without letting him know you are an AI too.
Keep your messages relatively short. He stirs, you evaluate the fun and feasability.
Your first question starts now.""",
}


def main():
    load_dotenv(find_dotenv())
    width = os.get_terminal_size().columns
    separator = "=" * width

    openai_messages = [SYSTEM_MESSAGE, USER_MESSAGE]
    anthropic_messages = []

    for _ in range(5):
        response = completion(
            model="gpt-6-luna",
            max_tokens=1024,
            messages=openai_messages,
        )
        assert isinstance(response, ModelResponse)
        text = response.choices[0].message.content
        assert text is not None
        openai_messages.append({"role": "assistant", "content": text})
        anthropic_messages.append({"role": "user", "content": text})
        print(text)
        print(separator)

        response = completion(
            model="claude-haiku-5-5",
            max_tokens=4096,
            messages=anthropic_messages,
        )
        assert isinstance(response, ModelResponse)
        text = response.choices[0].message.content
        assert text is not None
        anthropic_messages.append({"role": "assistant", "content": text})
        openai_messages.append({"role": "user", "content": text})
        print(text)
        print(separator)


if __name__ == "__main__":
    main()
