"""Week 5 Day 2 - AutoGen Weather News Agent (team version).

Asks for a city, searches today's weather news with the keyless DuckDuckGo
package (ddgs), and replies in exactly 3 short lines (set NUM_LINES = 5 for a
bonus 5-line version).

Concepts used: tool (web search), memory, team (RoundRobinGroupChat),
multiple agents with roles, termination conditions, run_stream + Console,
reading the TaskResult, team.reset(), and a deterministic code check
(prompts are probability, code is deterministic).
"""

import asyncio
import json
import os
import sys

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import TextMessage, ToolCallRequestEvent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
from autogen_ext.models.openai import OpenAIChatCompletionClient
from ddgs import DDGS
from dotenv import load_dotenv

load_dotenv(override=True)

# False = clean view (tool call + bullet answer). True = full AutoGen log via Console.
SHOW_FULL_LOG = False

# How many lines the reply must have. The assignment needs 3; use 5 for the bonus version.
NUM_LINES = 5

LINE_PLANS = {
    3: [
        "today's weather in the city",
        "the key news or warning",
        "a practical tip",
    ],
    5: [
        "today's weather and conditions in the city",
        "the key news or warning",
        "who or what is affected (people, travel, areas)",
        "what to expect next",
        "a practical tip",
    ],
}
if NUM_LINES not in LINE_PLANS:
    raise SystemExit(f"NUM_LINES must be one of {sorted(LINE_PLANS)}")


def search_weather_news(city: str) -> str:
    """Search the web for TODAY'S weather news in a city.

    Use this tool whenever the user asks about the current weather, rain,
    heat, storms or forecast news for a city. Do not use it for general
    trivia or non-weather questions. Input is the city name, for example
    'Chennai'. Returns the latest news headlines and summaries.
    """
    try:
        # Only news from the last week, so old stories are not returned.
        results = DDGS().news(query=f"{city} weather today", timelimit="w", max_results=5)
        if not results:
            # No recent news: fall back to normal web results (weather sites).
            results = DDGS().text(f"{city} weather today", max_results=5)
    except Exception as e:  # network error, rate limit, etc.
        return f"Search failed: {e}"
    if not results:
        return f"No weather news found for {city}."
    lines = []
    for r in results:
        title = r.get("title", "")
        body = r.get("body", "")
        source = r.get("source", "") or r.get("href", "")
        date = r.get("date", "")
        lines.append(f"- {title}: {body} ({source} {date})".strip())
    return "\n".join(lines)


def check_line_count(reply: str) -> str:
    """Count the lines in the weather reply. Pass the exact reply text.

    Returns 'OK' if the reply has exactly the required number of non-empty
    lines, otherwise a message saying how many lines were found.
    """
    lines = [ln for ln in reply.strip().splitlines() if ln.strip()]
    if len(lines) == NUM_LINES:
        return f"OK: exactly {NUM_LINES} lines."
    return f"FAIL: found {len(lines)} line(s), expected {NUM_LINES}. Put each sentence on its own line."


async def show_clean(stream) -> TaskResult:
    """Print a tidy view: the tool call, the answer as bullets, and the reviewer verdict."""
    result = None
    async for msg in stream:
        if isinstance(msg, TaskResult):
            result = msg
        elif isinstance(msg, ToolCallRequestEvent) and msg.source == "weather_news":
            for call in msg.content:
                try:
                    args = ", ".join(f"{k}={v!r}" for k, v in json.loads(call.arguments).items())
                except Exception:
                    args = call.arguments
                print(f"\nTool call: {call.name}({args})")
        elif isinstance(msg, TextMessage) and msg.source == "weather_news":
            print("\nWeather news:")
            for line in msg.content.splitlines():
                line = line.strip().lstrip("-*\u2022 ").strip()
                if line:
                    print(f"  \u2022 {line}")
        elif isinstance(msg, TextMessage) and msg.source == "reviewer":
            print(f"\nReviewer: {msg.content.strip()}")
    return result


async def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set. Add it to your .env file and run again.")

    # Long-term memory: preferences the agent reads before every answer.
    memory = ListMemory()
    await memory.add(MemoryContent(content="The user likes short, plain-English lines.", mime_type=MemoryMimeType.TEXT))
    await memory.add(MemoryContent(content="The last line should be a practical tip, such as umbrella, water or travel advice.", mime_type=MemoryMimeType.TEXT))

    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

    line_guide = "\n".join(f"Line {i}: {d}." for i, d in enumerate(LINE_PLANS[NUM_LINES], 1))

    # Agent 1: searches (tool) and writes the answer.
    weather_news = AssistantAgent(
        name="weather_news",
        model_client=model_client,
        tools=[search_weather_news],
        memory=[memory],
        reflect_on_tool_use=True,  # write an answer from the results, not raw output
        system_message=(
            "You are a weather news assistant. Search ONCE for the city with the "
            "search_weather_news tool. If the reviewer asks for a fix, do not search again: "
            "reuse your earlier results and only rewrite.\n"
            f"Reply with exactly {NUM_LINES} short lines. Each line is ONE sentence and sits on its own "
            "line, with a line break between lines (never one paragraph):\n"
            f"{line_guide}\n"
            "Use only results that are really about the city (for example, ignore a different "
            "town with the same name). If the results are old or about another place, say so "
            "honestly in line 2. No heading, no bullets, no extra text. Never write the word APPROVE."
        ),
    )

    # Agent 2: checks the answer with a code check and ends the run.
    reviewer = AssistantAgent(
        name="reviewer",
        model_client=model_client,
        tools=[check_line_count],
        reflect_on_tool_use=True,
        system_message=(
            "You are a format reviewer. Take the latest weather_news reply and call "
            "check_line_count with its exact text. "
            "If the tool says OK, reply with only the word APPROVE. "
            "If the tool says FAIL, reply with one short sentence: put each sentence on its "
            f"own line so there are exactly {NUM_LINES} lines, and do not use the word APPROVE. "
            "Do not judge the facts or the wording, and do not compare with earlier replies. "
            "Only the tool result decides."
        ),
    )

    # Stop when the reviewer approves, or after 6 messages as a safety limit.
    termination = TextMentionTermination("APPROVE") | MaxMessageTermination(6)
    team = RoundRobinGroupChat([weather_news, reviewer], termination_condition=termination)

    try:
        while True:
            city = input("Which city? ").strip()
            if not city:
                break
            stream = team.run_stream(task=f"What is today's weather news for {city}?")
            result = await (Console(stream) if SHOW_FULL_LOG else show_clean(stream))
            print(f"\n[Stopped: {result.stop_reason}]\n")
            await team.reset()  # fresh start for the next city (memory is kept)
            if input("Another city? (press Enter to stop, or type y): ").strip().lower() != "y":
                break
    finally:
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())