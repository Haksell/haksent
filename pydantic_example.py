from dotenv import find_dotenv, load_dotenv
from litellm import completion
from litellm.types.utils import ModelResponse
from pydantic import BaseModel


class ExtractedInfo(BaseModel):
    name: str
    email: str
    phone: str | None = None
    job: str | None = None


def main():
    load_dotenv(find_dotenv())

    messages = [
        {
            "role": "user",
            "content": "My name is John Smith, my email is john@example.com, and my phone is 555-1234.",
        },
    ]
    response = completion(
        model="gpt-6-luna",
        max_tokens=2048,
        messages=messages,
        response_format=ExtractedInfo,
    )
    assert isinstance(response, ModelResponse)
    text = response.choices[0].message.content
    assert text is not None
    print(text)


if __name__ == "__main__":
    main()
