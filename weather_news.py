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
import re
import sys
import urllib.parse
import urllib.request
from datetime import datetime

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

# True = also save the reply as a weather card picture (PNG). Needs: pip install pillow
SAVE_IMAGE = True

LINE_PLANS = {
    3: [
        "today's weather in the city with the current temperature in degrees C (from get_temperature)",
        "the key news or warning",
        "a practical tip",
    ],
    5: [
        "today's weather in the city with the current temperature in degrees C (from get_temperature)",
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


# Latest numbers from get_temperature, used to draw the picture card.
LAST_WEATHER = {}

WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Fog", 51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle",
    56: "Freezing drizzle", 57: "Freezing drizzle", 61: "Light rain", 63: "Rain", 65: "Heavy rain",
    66: "Freezing rain", 67: "Freezing rain", 71: "Light snow", 73: "Snow", 75: "Heavy snow",
    77: "Snow grains", 80: "Rain showers", 81: "Rain showers", 82: "Violent rain showers",
    85: "Snow showers", 86: "Snow showers", 95: "Thunderstorm", 96: "Thunderstorm with hail",
    99: "Thunderstorm with hail",
}


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_temperature(city: str) -> str:
    """Get the CURRENT temperature and conditions for a city, in degrees Celsius.

    Use this tool for the real temperature, feels-like temperature, humidity,
    wind, today's high and low, and rain chance. Input is the city name,
    for example 'Chennai'. Always use it when the answer needs a temperature.
    """
    try:
        geo = _get_json(
            "https://geocoding-api.open-meteo.com/v1/search?"
            + urllib.parse.urlencode({"name": city, "count": 1, "language": "en", "format": "json"})
        )
        places = geo.get("results") or []
        if not places:
            return f"Could not find the city '{city}'."
        place = places[0]
        data = _get_json(
            "https://api.open-meteo.com/v1/forecast?"
            + urllib.parse.urlencode({
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "timezone": "auto",
                "forecast_days": 1,
            })
        )
        cur, day = data["current"], data["daily"]
        info = {
            "place": f"{place['name']}, {place.get('country', '')}".strip(", "),
            "temp": round(cur["temperature_2m"]),
            "feels": round(cur["apparent_temperature"]),
            "humidity": cur["relative_humidity_2m"],
            "wind": round(cur["wind_speed_10m"]),
            "condition": WMO_CODES.get(cur["weather_code"], "Unknown"),
            "high": round(day["temperature_2m_max"][0]),
            "low": round(day["temperature_2m_min"][0]),
            "rain_chance": (day.get("precipitation_probability_max") or [None])[0],
        }
    except Exception as e:  # network error, bad response, etc.
        return f"Temperature lookup failed: {e}"
    LAST_WEATHER.clear()
    LAST_WEATHER.update(info)
    rain = f", rain chance {info['rain_chance']}%" if info["rain_chance"] is not None else ""
    return (
        f"{info['place']}: {info['temp']}\u00b0C now (feels like {info['feels']}\u00b0C), "
        f"{info['condition']}, humidity {info['humidity']}%, wind {info['wind']} km/h, "
        f"today's high {info['high']}\u00b0C and low {info['low']}\u00b0C{rain}."
    )


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


def _load_font(names, size):
    """Try a list of font files (Windows first, then Linux) and fall back to the default."""
    from PIL import ImageFont

    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap(draw, text, font, max_width):
    """Split text into lines that fit max_width pixels."""
    words, lines, current = text.split(), [], ""
    for word in words:
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=font) <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def make_weather_card(city: str, lines: list, path: str) -> str:
    """Draw the weather reply as a portrait picture (WhatsApp / LinkedIn friendly)."""
    from PIL import Image, ImageDraw

    regular = ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"]
    bold = ["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"]
    f_city, f_date = _load_font(bold, 80), _load_font(regular, 32)
    f_text, f_num = _load_font(regular, 36), _load_font(bold, 34)
    f_foot = _load_font(regular, 26)

    all_text = " ".join(lines).lower()
    if any(w in all_text for w in ("rain", "storm", "flood", "shower", "cyclone", "thunder")):
        top, bottom, icon = (52, 84, 140), (12, 20, 42), "rain"
    elif any(w in all_text for w in ("sunny", "hot", "heat", "clear", "warm")):
        top, bottom, icon = (245, 158, 11), (146, 64, 14), "sun"
    else:
        top, bottom, icon = (59, 98, 176), (15, 27, 51), "cloud"

    W, pad, card_pad, badge = 1080, 60, 30, 64
    probe = ImageDraw.Draw(Image.new("RGB", (W, 10)))
    text_w = W - 2 * pad - 2 * card_pad - badge - 24
    wrapped = [_wrap(probe, ln, f_text, text_w) for ln in lines]
    line_h = 48
    card_hs = [max(badge, len(w) * line_h) + 2 * card_pad for w in wrapped]
    header_h = 420
    H = header_h + sum(card_hs) + 24 * len(card_hs) + 130

    img = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(img)
    for y in range(H):  # vertical gradient
        t = y / H
        d.line([(0, y), (W, y)], fill=tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3)))

    # header
    d.text((pad, 70), city.title(), font=f_city, fill="white")
    d.text((pad, 175), datetime.now().strftime("%A, %d %B %Y"), font=f_date, fill=(220, 230, 250))
    w = LAST_WEATHER
    sub = "Today's weather news"
    if w:
        sub = f"{w['condition']}  |  High {w['high']}\u00b0C  Low {w['low']}\u00b0C"
    d.text((pad, 235), sub, font=f_date, fill=(255, 255, 255))
    if w:
        d.text((pad, 290), f"{w['temp']}\u00b0C", font=_load_font(bold, 92), fill="white")
        d.text((pad + 300, 312), f"Feels like {w['feels']}\u00b0C  |  Humidity {w['humidity']}%", font=f_date, fill=(235, 240, 250))
        d.text((pad + 300, 356), f"Wind {w['wind']} km/h", font=f_date, fill=(235, 240, 250))

    # icon (top right)
    cx, cy = W - 190, 150
    if icon == "sun":
        d.ellipse([cx - 55, cy - 55, cx + 55, cy + 55], fill=(255, 236, 140))
        for k in range(8):
            import math

            a = k * math.pi / 4
            d.line([(cx + 78 * math.cos(a), cy + 78 * math.sin(a)), (cx + 105 * math.cos(a), cy + 105 * math.sin(a))], fill=(255, 236, 140), width=8)
    else:
        cloud = (235, 240, 250)
        d.ellipse([cx - 110, cy - 20, cx - 20, cy + 60], fill=cloud)
        d.ellipse([cx - 60, cy - 70, cx + 40, cy + 40], fill=cloud)
        d.ellipse([cx, cy - 30, cx + 110, cy + 60], fill=cloud)
        d.rounded_rectangle([cx - 100, cy + 10, cx + 100, cy + 60], radius=25, fill=cloud)
        if icon == "rain":
            for dx in (-60, -15, 30, 75):
                d.line([(cx + dx, cy + 80), (cx + dx - 14, cy + 120)], fill=(150, 200, 255), width=7)

    # cards
    y = header_h
    for n, (wrap_lines, ch) in enumerate(zip(wrapped, card_hs), 1):
        d.rounded_rectangle([pad, y, W - pad, y + ch], radius=28, fill=(255, 255, 255))
        by = y + (ch - badge) // 2
        d.ellipse([pad + card_pad, by, pad + card_pad + badge, by + badge], fill=top)
        nw = d.textlength(str(n), font=f_num)
        d.text((pad + card_pad + (badge - nw) / 2, by + 12), str(n), font=f_num, fill="white")
        ty = y + (ch - len(wrap_lines) * line_h) // 2
        for ln in wrap_lines:
            d.text((pad + card_pad + badge + 24, ty), ln, font=f_text, fill=(28, 35, 51))
            ty += line_h
        y += ch + 24

    d.text((pad, H - 90), "Sources: DuckDuckGo News, Open-Meteo  |  AutoGen agent", font=f_foot, fill=(210, 220, 240))
    img.save(path)
    return path


def answer_lines(result) -> list:
    """Return the weather_news agent's latest answer as clean lines (no bullets)."""
    for msg in reversed(result.messages):
        if isinstance(msg, TextMessage) and msg.source == "weather_news":
            cleaned = [ln.strip().lstrip("-*\u2022 ").strip() for ln in msg.content.splitlines()]
            return [ln for ln in cleaned if ln]
    return []


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
        tools=[search_weather_news, get_temperature],
        memory=[memory],
        reflect_on_tool_use=True,  # write an answer from the results, not raw output
        system_message=(
            "You are a weather news assistant. For the city given, call get_temperature ONCE "
            "for the real temperature and search_weather_news ONCE for the news. If the reviewer "
            "asks for a fix, do not call tools again: reuse your earlier results and only rewrite.\n"
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
            LAST_WEATHER.clear()
            stream = team.run_stream(task=f"What is today's weather news for {city}?")
            result = await (Console(stream) if SHOW_FULL_LOG else show_clean(stream))
            print(f"\n[Stopped: {result.stop_reason}]\n")
            if SAVE_IMAGE:
                lines = answer_lines(result)
                if lines:
                    try:
                        name = "weather_" + re.sub(r"[^a-z0-9]+", "_", city.lower()).strip("_") + ".png"
                        make_weather_card(city, lines, name)
                        print(f"Picture saved: {os.path.abspath(name)}")
                        if hasattr(os, "startfile"):  # Windows: open the picture
                            os.startfile(name)
                    except ImportError:
                        print("To save a picture, run: pip install pillow")
                    except Exception as e:
                        print(f"Could not make the picture: {e}")
            await team.reset()  # fresh start for the next city (memory is kept)
            if input("Another city? (press Enter to stop, or type y): ").strip().lower() != "y":
                break
    finally:
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())