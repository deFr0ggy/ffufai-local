#!/usr/bin/env python3

import argparse
import os
import subprocess
import requests
import json
from openai import OpenAI
import anthropic
from urllib.parse import urlparse
import tempfile
from bs4 import BeautifulSoup

DEFAULT_MODELS = {
    'openai': 'gpt-4o',
    'anthropic': 'claude-sonnet-4-20250514',
    'ollama': 'llama2',
    'lmstudio': 'llama2'
}


def get_provider_config(args):
    provider = args.model_provider
    openai_key = os.getenv('OPENAI_API_KEY')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')

    if provider:
        provider = provider.lower()
        if provider not in ['openai', 'anthropic', 'ollama', 'lmstudio']:
            raise ValueError("Unsupported provider. Use one of: openai, anthropic, ollama, lmstudio.")
    else:
        if openai_key:
            provider = 'openai'
        elif anthropic_key:
            provider = 'anthropic'
        else:
            raise ValueError("No provider selected and no OpenAI/Anthropic API key found. Use --model-provider to select ollama or lmstudio, or set OPENAI_API_KEY / ANTHROPIC_API_KEY.")

    if provider == 'openai':
        if not openai_key:
            raise ValueError("OPENAI_API_KEY is required for the OpenAI provider.")
        return provider, openai_key
    elif provider == 'anthropic':
        if not anthropic_key:
            raise ValueError("ANTHROPIC_API_KEY is required for the Anthropic provider.")
        return provider, anthropic_key
    else:
        return provider, None


def build_messages(system, user):
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]


def call_openai_chat(model, messages, api_key):
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content.strip()


def call_anthropic_chat(model, messages, api_key):
    client = anthropic.Anthropic(api_key=api_key)
    system = next((m['content'] for m in messages if m['role'] == 'system'), '')
    user = next((m['content'] for m in messages if m['role'] == 'user'), '')
    response = client.messages.create(
        model=model,
        max_tokens=10000,
        temperature=0,
        system=system,
        messages=[{"role": "user", "content": user}]
    )
    return response.content[0].text.strip()


def call_ollama_chat(model, messages, ollama_url):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0
    }
    response = requests.post(f"{ollama_url.rstrip('/')}/v1/chat/completions", json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data['choices'][0]['message']['content'].strip()


def call_lmstudio_chat(model, messages, lmstudio_url):
    url_base = lmstudio_url.rstrip('/')
    payloads = [
        {
            "url": f"{url_base}/api/prompt",
            "json": {
                "model": model,
                "input": messages,
                "parameters": {
                    "temperature": 0,
                    "max_new_tokens": 2000
                }
            }
        },
        {
            "url": f"{url_base}/api/v1/prompt",
            "json": {
                "model": model,
                "input": messages,
                "parameters": {
                    "temperature": 0,
                    "max_new_tokens": 2000
                }
            }
        },
        {
            "url": f"{url_base}/v1/chat/completions",
            "json": {
                "model": model,
                "messages": messages,
                "temperature": 0
            }
        },
        {
            "url": f"{url_base}/api/v1/chat/completions",
            "json": {
                "model": model,
                "messages": messages,
                "temperature": 0
            }
        },
        {
            "url": f"{url_base}/v1/completions",
            "json": {
                "model": model,
                "prompt": messages,
                "temperature": 0,
                "max_tokens": 2000
            }
        },
        {
            "url": f"{url_base}/api/v1/completions",
            "json": {
                "model": model,
                "prompt": messages,
                "temperature": 0,
                "max_tokens": 2000
            }
        }
    ]

    last_error = None
    for attempt in payloads:
        try:
            response = requests.post(attempt["url"], json=attempt["json"], timeout=120)
            if response.status_code == 404:
                last_error = f"404 from {attempt['url']}"
                continue
            if response.status_code >= 400:
                last_error = f"{response.status_code} from {attempt['url']}: {response.text.strip()}"
                continue
            data = response.json()
            if isinstance(data, dict) and 'response' in data:
                return data['response'].strip()
            if isinstance(data, dict) and 'output' in data:
                output = data['output']
                if isinstance(output, str):
                    return output.strip()
                if isinstance(output, dict) and 'text' in output:
                    return output['text'].strip()
                if isinstance(output, list) and output:
                    first = output[0]
                    if isinstance(first, dict) and 'text' in first:
                        return first['text'].strip()
            if isinstance(data, dict) and 'choices' in data:
                choice = data['choices'][0]
                if isinstance(choice, dict):
                    if isinstance(choice.get('message'), dict):
                        return choice.get('message', {}).get('content', '').strip()
                    if 'text' in choice:
                        return choice['text'].strip()
            if isinstance(data, str):
                return data.strip()
            last_error = f"Unable to parse response from {attempt['url']}: {data}"
        except requests.RequestException as e:
            last_error = f"Request failed for {attempt['url']}: {e}"
            continue

    raise ValueError(f"Unable to communicate with LMStudio server. Last error: {last_error}")


def send_completion(provider, model_name, api_key, ollama_url, lmstudio_url, system, user):
    model = model_name or DEFAULT_MODELS.get(provider)
    if provider == 'openai':
        return call_openai_chat(model, build_messages(system, user), api_key)
    elif provider == 'anthropic':
        return call_anthropic_chat(model, build_messages(system, user), api_key)
    elif provider == 'ollama':
        return call_ollama_chat(model, build_messages(system, user), ollama_url)
    elif provider == 'lmstudio':
        return call_lmstudio_chat(model, build_messages(system, user), lmstudio_url)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def get_response(url):
    try:
        response = requests.get(url, allow_redirects=True)

        soup = BeautifulSoup(response.content, 'html.parser')

        for tag in soup.select('style, link[rel="stylesheet"]'):
            tag.decompose()

        for tag in soup.find_all(True):
            if hasattr(tag, 'attrs') and tag.attrs is not None:
                tag.attrs.pop('style', None)

            if tag.name == 'svg':
                tag.decompose()
            if tag.name == 'img':
                tag.decompose()

        content = soup.prettify()

        return {
            "url": response.url,
            "headers": dict(response.headers),
            "cookies": dict(response.cookies),
            "content": content[:2500]
        }

    except requests.RequestException as e:
        print(f"Error fetching content: {e}")
        return {"error": "Error fetching content."}

def get_headers(url):
    try:
        response = requests.head(url, allow_redirects=True)
        return dict(response.headers)
    except requests.RequestException as e:
        print(f"Error fetching headers: {e}")
        return {"Header": "Error fetching headers."}

def get_ai_extensions(url, headers, provider, api_key, model_name, ollama_url, lmstudio_url, max_extensions):
    prompt = f"""
    Given the following URL and HTTP headers, suggest the most likely file extensions for fuzzing this endpoint.
    Respond with a JSON object containing a list of extensions. The response will be parsed with json.loads(),
    so it must be valid JSON. No preamble or yapping. Use the format: {{"extensions": [".ext1", ".ext2", ...]}}.
    Do not suggest more than {max_extensions}, but only suggest extensions that make sense. For example, if the path is
    /js/ then don't suggest .css as the extension. Also, if limited, prefer the extensions which are more interesting.
    The URL path is great to look at for ideas. For example, if it says presentations, then it's likely there
    are powerpoints or pdfs in there. If the path is /js/ then it's good to use js as an extension.

    Examples:
    1. URL: https://example.com/presentations/FUZZ
       Headers: {{"Content-Type": "application/pdf", "Content-Length": "1234567"}}
       JSON Response: {{"extensions": [".pdf", ".ppt", ".pptx"]}}

    2. URL: https://example.com/FUZZ
       Headers: {{"Server": "Microsoft-IIS/10.0", "X-Powered-By": "ASP.NET"}}
       JSON Response: {{"extensions": [".aspx", ".asp", ".exe", ".dll"]}}

    URL: {url}
    Headers: {headers}

    JSON Response:
    """

    system = "You are a helpful assistant that suggests file extensions for fuzzing based on URL and headers."
    response_text = send_completion(
        provider,
        model_name,
        api_key,
        ollama_url,
        lmstudio_url,
        system,
        prompt
    )
    return json.loads(response_text.strip())

def get_contextual_wordlist(url, headers, provider, api_key, model_name, ollama_url, lmstudio_url, max_size, cookies=None, content=None):
    prompt = f"""
    Given the following URL and HTTP headers, suggest the most likely contextual wordlist for content discovery on this endpoint.
    Be as extensive as possible, provide the maximum number of directories and files that make sense for the endpoint.
    Try to create a list of size {max_size}.
    Respond with a JSON object containing a list of directories and files. The response will be parsed with json.loads(),
    so it must be valid JSON. No preamble or yapping. Use the format: {{"wordlist": ["dir1", "dir2", "file1", "file2"]}}.
    Only make suggestions that make sense. For example, if domain is for a book shop
    then don't suggest footbal as a directory. Also, if limited, prefer the files and directories which are more interesting.
    The URL path is great to look at for ideas, and so is the brand behind the URL.
    Focus on contents relevant to the identified industry and technology stack. Include technology-specific files.
    For example, if it says presentations, then it's likely there are powerpoints or pdfs in there. If the path is /js/ then it's good to fuzz for JS files.

    Example 1: WordPress Blog
    URL: https://blog.techstartup.io/wp-content/uploads/2024/FUZZ
    Headers: {{
      "Server": "nginx/1.22.1",
      "X-Powered-By": "PHP/8.1.2",
      "Link": "<https://blog.techstartup.io/wp-json/>; rel=\"https://api.w.org/\"",
      "Content-Type": "image/jpeg"
    }}

    Response:
    {{
      "wordlist": ["wp-content", "wp-includes", "wp-admin", "uploads", "themes", "plugins", "2024", "2023", "backup", "cache", "wp-config.php", "xmlrpc.php", "wp-login.php", "readme.html", ".htaccess", "wp-config.php.bak", "debug.log"]
    }}

    Example 2: E-commerce Platform
    URL: https://shop.globalretail.com/checkout/payment/FUZZ
    Headers:
    {{
      "Server": "Microsoft-IIS/10.0",
      "X-Powered-By": "ASP.NET",
      "X-AspNet-Version": "4.0.30319",
      "X-Frame-Options": "SAMEORIGIN",
      "Strict-Transport-Security": "max-age=31536000"
    }}

    Response:
    {{
      "wordlist": ["checkout", "payment", "api", "admin", "account", "orders", "products", "cart", "invoice", "App_Data", "bin", "Content", "web.config", "Global.asax", "payment.aspx", "checkout.aspx", "web.config.bak", "App_Data.mdf", "connectionstrings.config"]
    }}

    URL: {url}
    Headers: {headers}
    Cookies: {cookies}
    Content: {content}

    JSON Response:
    """

    system = "You are a helpful assistant that suggests wordlists for fuzzing based on URL and headers."
    response_text = send_completion(
        provider,
        model_name,
        api_key,
        ollama_url,
        lmstudio_url,
        system,
        prompt
    )
    return json.loads(response_text.strip())

def main():
    parser = argparse.ArgumentParser(description='ffufai - AI-powered ffuf wrapper')
    parser.add_argument('--ffuf-path', default='ffuf', help='Path to ffuf executable')
    parser.add_argument('--max-extensions', type=int, default=4, help='Maximum number of extensions to suggest')
    parser.add_argument('--wordlists', action='store_true', help='Generate contextual wordlists')
    parser.add_argument('--max-wordlist-size', type=int, help="The maximum size of the generated wordlist")
    parser.add_argument('--include-response', action='store_true', help='Makes a GET request and uses the Response as context for better wordlist generation (Uses more tokens)')
    parser.add_argument('--model-provider', choices=['openai', 'anthropic', 'ollama', 'lmstudio'], help='AI provider to use')
    parser.add_argument('--model-name', help='Model name for the selected provider')
    parser.add_argument('--ollama-url', default='http://127.0.0.1:11434', help='Base URL for the local Ollama server')
    parser.add_argument('--lmstudio-url', default='http://127.0.0.1:8080', help='Base URL for the local LMStudio server')
    args, unknown = parser.parse_known_args()

    try:
        url_index = unknown.index('-u') + 1
        url = unknown[url_index]
    except (ValueError, IndexError):
        print("Error: -u URL argument is required.")
        return

    parsed_url = urlparse(url)
    path_parts = parsed_url.path.split('/')
    base_url = url.replace('FUZZ', '')

    if 'FUZZ' not in path_parts[-1]:
        print("Warning: FUZZ keyword is not at the end of the URL path. Extension fuzzing may not work as expected.")

    headers = get_headers(base_url)

    try:
        provider, api_key = get_provider_config(args)
    except ValueError as e:
        print(f"Error: {e}")
        return

    model_name = args.model_name or DEFAULT_MODELS.get(provider)

    if args.wordlists:
        try:
            if args.max_wordlist_size:
                size = args.max_wordlist_size
            else:
                size = 200

            if args.include_response:
                response = get_response(base_url)
                headers = response['headers']
                cookies = response['cookies']
                content = response['content']
                wordlists_data = get_contextual_wordlist(
                    url,
                    headers,
                    provider,
                    api_key,
                    model_name,
                    args.ollama_url,
                    args.lmstudio_url,
                    size,
                    cookies=cookies,
                    content=content
                )
            else:
                wordlists_data = get_contextual_wordlist(
                    url,
                    headers,
                    provider,
                    api_key,
                    model_name,
                    args.ollama_url,
                    args.lmstudio_url,
                    size
                )

            print(wordlists_data)
            wordlist = '\n'.join(wordlists_data['wordlist'])

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing AI response. The Wordlist size may have been too big for your max_tokens. Try again. Error: {e}")
            return

        if wordlist:
            file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
            file.write(wordlist)
            file.close()
            ffuf_command = [args.ffuf_path] + unknown + ['-w', file.name]
            subprocess.run(ffuf_command)


    else:
        try:
            extensions_data = get_ai_extensions(
                url,
                headers,
                provider,
                api_key,
                model_name,
                args.ollama_url,
                args.lmstudio_url,
                args.max_extensions
            )
            print(extensions_data)
            extensions = ','.join(extensions_data['extensions'][:args.max_extensions])

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing AI response. Try again. Error: {e}")
            return

        ffuf_command = [args.ffuf_path] + unknown + ['-e', extensions]

        subprocess.run(ffuf_command)


if __name__ == '__main__':
    main()
