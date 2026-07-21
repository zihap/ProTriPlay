import os
from openai import OpenAI
from volcenginesdkarkruntime import Ark
from config import (
    ark_api_key,
    ark_base_url,
    ark_model,
    openai_api_key,
    openai_base_url,
    deepseek_api_key,
    deepseek_base_url,
    qwen_api_key,
    qwen_base_url,
    http_proxy,
    https_proxy,
    use_model,
)

os.environ["http_proxy"] = http_proxy
os.environ["https_proxy"] = https_proxy


def parse_ark_response(response):
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text

    if hasattr(response, "output") and isinstance(response.output, list):
        for output_item in response.output:
            if hasattr(output_item, "type") and output_item.type == "message":
                if hasattr(output_item, "content") and isinstance(
                    output_item.content, list
                ):
                    for content_item in output_item.content:
                        if (
                            hasattr(content_item, "type")
                            and content_item.type == "output_text"
                        ):
                            if hasattr(content_item, "text") and content_item.text:
                                return content_item.text
                            if (
                                hasattr(content_item, "content")
                                and content_item.content
                            ):
                                return content_item.content

            if hasattr(output_item, "type") and output_item.type == "output_text":
                if hasattr(output_item, "text") and output_item.text:
                    return output_item.text
                if hasattr(output_item, "content") and output_item.content:
                    return output_item.content

    return str(response)


def handle_stream_response(client, model, messages, extra_body=None):
    if extra_body is None:
        extra_body = {}

    if use_model == "ark":
        response = client.responses.create(
            model=ark_model, input=messages, extra_body=extra_body
        )
        return parse_ark_response(response)
    elif use_model == "qwen":
        extra_body["enable_thinking"] = False

        response_stream = client.chat.completions.create(
            model=use_model, messages=messages, stream=True, extra_body=extra_body
        )
        response_content = ""
        for chunk in response_stream:
            if (
                chunk.choices
                and chunk.choices[0].delta
                and chunk.choices[0].delta.content
            ):
                response_content += chunk.choices[0].delta.content
        return response_content
    else:
        response = client.chat.completions.create(
            model=use_model, messages=messages, extra_body=extra_body
        )
        return response.choices[0].message.content


def get_ark_client():
    return Ark(api_key=ark_api_key, base_url=ark_base_url, timeout=1800, max_retries=2)


def get_client():
    if use_model == "ark":
        return get_ark_client()
    elif use_model == "gpt":
        return OpenAI(api_key=openai_api_key, base_url=openai_base_url)
    elif use_model == "deepseek":
        return OpenAI(api_key=deepseek_api_key, base_url=deepseek_base_url)
    elif use_model == "qwen":
        return OpenAI(api_key=qwen_api_key, base_url=qwen_base_url)
    return get_ark_client()
