import asyncio
import json

from dotenv import find_dotenv, load_dotenv
from litellm import ModelResponse, acompletion
from pydantic import BaseModel

semaphore = asyncio.Semaphore(3)


class ExtractedInfo(BaseModel):
    result: int


async def get_response(prompt: str) -> str:
    async with semaphore:
        response = await acompletion(
            model="gpt-6-luna",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            response_format=ExtractedInfo,
            num_retries=3,
        )
        assert isinstance(response, ModelResponse)
        text = response.choices[0].message.content
        assert text is not None
        res = json.loads(text)
        return res["result"]


async def main():
    load_dotenv(find_dotenv())

    prompts = [f"What is {i} + {i}?" for i in range(5)]
    tasks = list(map(get_response, prompts))
    results = await asyncio.gather(*tasks)
    for i, (prompt, result) in enumerate(zip(prompts, results)):
        print(f"Q: {prompt}")
        print(f"A: {result}")
        print(i + i == result)
        print()


if __name__ == "__main__":
    asyncio.run(main())
