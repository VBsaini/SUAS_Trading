"""ResearchLab API: question interpretation and validated backtests."""

from __future__ import annotations

import json
import os
import re
from datetime import date
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
from flask import Flask, jsonify, request
from dotenv import load_dotenv

from backtest import load_ohlc_data, run_backtest

load_dotenv()
app = Flask(__name__)
DEFAULT_EXPERIMENT = {
	"market": "NIFTY 50",
	"threshold_percent": 3.0,
	"holding_period": 5,
	"entry_timing": "next_open",
	"exit_price": "close",
	"cost_percent": 0.1,
	"start_date": "2015-01-01",
	"end_date": "2025-12-31",
	"max_fall_percent": None,
	"min_fall_percent": None,
}
DATA_PATHS = {
	"NIFTY 50": os.getenv("NIFTY50_DATA_CSV_PATH", os.getenv("DATA_CSV_PATH", "data/nifty50_ohlc.csv")),
	"NIFTY 250": os.getenv("NIFTY250_DATA_CSV_PATH", "data/nifty250_ohlc.csv"),
}
AI_PROVIDER = os.getenv("AI_PROVIDER", "gemini").lower()
AI_MODEL = os.getenv("AI_MODEL", "gemini-3.6-flash")
AI_URL = os.getenv("AI_API_URL", "http://localhost:11434/api/chat")
CORS_ORIGINS = {
	origin.strip()
	for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
	if origin.strip()
}


@app.after_request
def add_cors_headers(response):
	origin = request.headers.get("Origin")
	if origin in CORS_ORIGINS:
		response.headers["Access-Control-Allow-Origin"] = origin
		response.headers["Access-Control-Allow-Headers"] = "Content-Type"
		response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
		response.headers["Vary"] = "Origin"
	return response


class RequestError(ValueError):
	pass


def _validate_experiment(value: object) -> dict[str, object]:
	if not isinstance(value, dict):
		exec('raise RequestError("experiment must be an object")')
	experiment = {**DEFAULT_EXPERIMENT, **value}
	try:
		experiment["threshold_percent"] = float(experiment["threshold_percent"])
		experiment["holding_period"] = int(experiment["holding_period"])
		experiment["cost_percent"] = float(experiment["cost_percent"])
		start = date.fromisoformat(str(experiment["start_date"]))
		end = date.fromisoformat(str(experiment["end_date"]))
	except (KeyError, TypeError, ValueError) as exc:
		raise RequestError("experiment contains invalid numeric or date fields") from exc
	if experiment["market"] not in DATA_PATHS:
		raise RequestError(f"market must be one of: {', '.join(DATA_PATHS)}")
	if experiment["threshold_percent"] <= 0 or experiment["holding_period"] < 1:
		raise RequestError("threshold_percent must be positive and holding_period must be at least 1")
	if experiment["cost_percent"] < 0:
		raise RequestError("cost_percent cannot be negative")
	if experiment["entry_timing"] not in {"next_open", "next_close"}:
		raise RequestError("entry_timing must be next_open or next_close")
	if experiment["exit_price"] != "close" or start > end:
		raise RequestError("exit_price must be close and start_date must be before end_date")
	# Validate optional fall filters
	max_fall = experiment.get("max_fall_percent")
	min_fall = experiment.get("min_fall_percent")
	if max_fall is not None:
		try:
			max_fall = float(max_fall)
			experiment["max_fall_percent"] = max_fall
		except (TypeError, ValueError) as exc:
			raise RequestError("max_fall_percent must be a number") from exc
		if max_fall <= 0:
			raise RequestError("max_fall_percent must be positive")
		if max_fall <= experiment["threshold_percent"]:
			raise RequestError("max_fall_percent must be greater than threshold_percent")
	if min_fall is not None:
		try:
			min_fall = float(min_fall)
			experiment["min_fall_percent"] = min_fall
		except (TypeError, ValueError) as exc:
			raise RequestError("min_fall_percent must be a number") from exc
		if min_fall <= 0:
			raise RequestError("min_fall_percent must be positive")
		if min_fall < experiment["threshold_percent"]:
			raise RequestError("min_fall_percent must be at least threshold_percent")
	return experiment


def _fallback_interpretation(question: str, current_experiment: dict[str, object] | None = None) -> dict[str, object]:
	lower_question = question.lower()
	base = {**DEFAULT_EXPERIMENT, **(current_experiment or {})}
	market = "NIFTY 250" if "250" in lower_question or "smallcap" in lower_question else str(base["market"])
	# Detect "work to be at least X%" / "work means X% increase" etc.
	work_threshold_match = re.search(
		r"work\s+(?:to\s+be\s+|as\s+|means?|s+|is\s+)?(?:at\s+least\s+|atleast\s+|>=?\s+)?(\d+(?:\.\d+)?)\s*%",
		lower_question,
	)
	# Detect fall range constraints:
	# - "fall not more than X%" / "fall more than X%" / "at most X% fall" → max_fall_percent
	# - "fall between X% and Y%" / "fall from X% to Y%" → min_fall_percent + threshold
	max_fall_match = re.search(
		r"(?:fall|falls|drop|drops|decline|declines|negative|down)\s+(?:not\s+more\s+than|no\s+more\s+than|at\s+most|up\s+to|max(?:imum)?\s+(?:of\s+)?|less\s+than|<|<=|within\s+(?:a\s+range\s+(?:of\s+)?)?)\s*(\d+(?:\.\d+)?)\s*%",
		lower_question,
	)
	# "do not consider ... fall ... more than X%" pattern
	if not max_fall_match:
		max_fall_match = re.search(
			r"do\s+not\s+consider\s+(?:cases\s+)?(?:when\s+|if\s+|where\s+)?(?:the\s+)?(?:fall|falls|drop|drops|decline|declines)\s+(?:is\s+|are\s+|was\s+|were\s+)?(?:more\s+than|above|beyond|exceed(?:ing)?|greater\s+than|>)\s*(\d+(?:\.\d+)?)\s*%",
			lower_question,
		)
	# "only consider ... fall ... at most X%" / "ignore ... fall ... more than X%"
	if not max_fall_match:
		max_fall_match = re.search(
			r"(?:only\s+consider|ignore|exclude|skip|filter\s+out)\s+(?:cases?\s+)?(?:when\s+|if\s+|where\s+)?(?:the\s+)?(?:fall|falls|drop|drops|decline|declines)\s+(?:is\s+|are\s+|was\s+|were\s+)?(?:of\s+)?(?:at\s+most|up\s+to|no\s+more\s+than|max(?:imum)?\s+(?:of\s+)?|less\s+than|within)\s*(\d+(?:\.\d+)?)\s*%",
			lower_question,
		)
	# Detect "fall between X% and Y%" → min_fall_percent = X, threshold_percent = Y
	between_match = re.search(
		r"(?:fall|falls|drop|drops|decline|declines)\s+(?:between|from)\s+(\d+(?:\.\d+)?)\s*%\s*(?:and|to|--|–|—)\s*(\d+(?:\.\d+)?)\s*%",
		lower_question,
	)
	threshold_match = re.search(r"(\d+(?:\.\d+)?)\s*%", lower_question)
	holding_match = re.search(r"(\d+)\s*(?:trading\s*)?(?:days?|sessions?)", lower_question)
	threshold = float(work_threshold_match.group(1)) if work_threshold_match else float(threshold_match.group(1)) if threshold_match else float(base["threshold_percent"])
	holding_period = int(holding_match.group(1)) if holding_match else int(base["holding_period"])
	is_rise = any(word in lower_question for word in ("rise", "rally", "gain", "up", "breakout"))
	direction = "rise" if is_rise else "fall"
	entry = "next_close" if "close" in lower_question and "open" not in lower_question else "next_open" if "open" in lower_question else str(base["entry_timing"])
	experiment = {**base, "market": market, "threshold_percent": threshold, "holding_period": holding_period, "entry_timing": entry}
	# Apply fall range filters detected from the question
	if max_fall_match:
		experiment["max_fall_percent"] = float(max_fall_match.group(1))
	elif between_match:
		experiment["min_fall_percent"] = float(between_match.group(1))
		experiment["threshold_percent"] = float(between_match.group(2))
	interpretation = f"Test whether {market} has positive forward returns after a one-day close-to-close {direction} of at least {threshold:g}%."
	critical_questions = []
	if not threshold_match and not work_threshold_match:
		critical_questions.append("What should count as a 'sharp fall': 1%, 3%, 5%, or another threshold?")
	return {
		"interpretation": interpretation,
		"assumptions": [
			"The signal is detected after the daily close.",
			f"Entry occurs at the next trading day's {'close' if entry == 'next_close' else 'open'}.",
			"The historical sample is a research proxy, not a forecast.",
			"Taking NIFTY 50 index.",
			f"Each trade is held for {holding_period} trading days.",
			f"A transaction cost of {experiment['cost_percent']:.2f}% is deducted from each trade's return.",
			f"The test period runs from {experiment['start_date']} to {experiment['end_date']}.",
			"Work is defined as making any profit (positive net return).",
		],
		"critical_questions": critical_questions,
		"hypothesis": f"A {direction} of at least {threshold:g}% in {market} produces positive average {holding_period}-day forward returns.",
		"suggested_experiment": experiment,
		"ai_generated": False,
		"question": question,
	}


def _ai_json(prompt: str) -> dict[str, object] | None:
	api_key = os.getenv("AI_API_KEY") or os.getenv("GEMINI_API_KEY")
	if AI_PROVIDER == "gemini":
		if not api_key:
			return None
		url = f"https://generativelanguage.googleapis.com/v1beta/models/{AI_MODEL}:generateContent?key={api_key}"
		payload = {"contents": [{"parts": [{"text": f"Return valid JSON only.\n{prompt}"}]}], "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"}}
	elif AI_PROVIDER == "ollama":
		url = AI_URL
		payload = {"model": AI_MODEL, "stream": False, "format": "json", "messages": [{"role": "user", "content": f"Return valid JSON only.\n{prompt}"}]}
	else:
		if not api_key:
			return None
		url = os.getenv("AI_API_URL", "https://api.openai.com/v1/chat/completions")
		payload = {"model": AI_MODEL, "temperature": 0.1, "response_format": {"type": "json_object"}, "messages": [{"role": "system", "content": "Return valid JSON only."}, {"role": "user", "content": prompt}]}
	def _extract_json(text: str) -> dict[str, object]:
		"""Pull a JSON object out of model output even when wrapped in markdown fences or surrounded by other text."""
		text = text.strip()
		if text.startswith("```"):
			text = re.sub(r"^```(?:json)?\s*", "", text, count=1)
			text = text.rsplit("```", 1)[0]
		text = text.strip()
		start = text.find("{")
		end = text.rfind("}")
		if start != -1 and end != -1 and end > start:
			text = text[start : end + 1]
		return json.loads(text)

	def _call_ollama_once(payload: dict) -> dict[str, object] | None:
		body = json.dumps(payload).encode()
		headers = {"Content-Type": "application/json"}
		response = urlopen(Request(url, data=body, headers=headers), timeout=20)
		content = json.loads(response.read())
		text = content["message"]["content"]
		try:
			return _extract_json(text)
		except (ValueError, json.JSONDecodeError):
			return None

	try:
		body = json.dumps(payload).encode()
		headers = {"Content-Type": "application/json"}
		if AI_PROVIDER != "gemini":
			headers["Authorization"] = f"Bearer {api_key}" if api_key else ""
		response = urlopen(Request(url, data=body, headers=headers), timeout=20)
		content = json.loads(response.read())
		if AI_PROVIDER == "gemini":
			text = content["candidates"][0]["content"]["parts"][0]["text"]
		elif AI_PROVIDER == "ollama":
			text = content["message"]["content"]
		else:
			text = content["choices"][0]["message"]["content"]
		# Try direct parse, then markdown-strip, then retry the API call with extraction
		for attempt in range(2):
			try:
				return _extract_json(text)
			except (ValueError, json.JSONDecodeError):
				if attempt == 0:
					response = urlopen(Request(url, data=body, headers=headers), timeout=20)
					content = json.loads(response.read())
					if AI_PROVIDER == "gemini":
						text = content["candidates"][0]["content"]["parts"][0]["text"]
					elif AI_PROVIDER == "ollama":
						text = content["message"]["content"]
					else:
						text = content["choices"][0]["message"]["content"]
					continue
				return None
	except (HTTPError, URLError, TimeoutError, KeyError, TypeError, ValueError, json.JSONDecodeError):
		return None


def interpret_question(
	question: str,
	current_experiment: dict[str, object] | None = None,
	previous_critical_questions: list[str] | None = None,
) -> dict[str, object]:
	fallback = _fallback_interpretation(question, current_experiment)
	context = json.dumps(current_experiment) if current_experiment else "none"
	previous_cq = json.dumps(previous_critical_questions) if previous_critical_questions else "none"
	generated = _ai_json(
		"Interpret this market research question. Return ONLY a JSON object with these "
		"exact keys: interpretation (string), assumptions (array of strings), "
		"critical_questions (array of strings), hypothesis (string), "
		"suggested_experiment (object).\n\n"
		"suggested_experiment MUST use exactly these fields and types:\n"
		"  market: string, one of 'NIFTY 50' or 'NIFTY 250'\n"
		"  threshold_percent: number (percent, e.g. 3.0)\n"
		"  holding_period: integer (trading days)\n"
		"  entry_timing: string, one of 'next_open' or 'next_close'\n"
		"  exit_price: string, always 'close'\n"
		"  cost_percent: number (percent, e.g. 0.1)\n"
		"  start_date: string in YYYY-MM-DD format\n"
		"  end_date: string in YYYY-MM-DD format\n"
		"  max_fall_percent: number or null (optional upper bound on fall magnitude)\n"
		"  min_fall_percent: number or null (optional lower bound on fall magnitude)\n\n"
		"Here is an example of the exact output shape for a different question:\n"
		'{\n'
		'  "interpretation": "Test whether NIFTY 50 has positive forward returns after a one-day close-to-close fall of at least 3%.",\n'
		'  "assumptions": ["The signal is detected after the daily close.", "Entry occurs at the next trading day\'s open.", "The historical sample is a research proxy, not a forecast.", "The market used is NIFTY 50.", "Each trade is held for 5 trading days.", "A transaction cost of 0.10% is deducted from each trade\'s return.", "The test period runs from 2015-01-01 to 2025-12-31."],\n'
		'  "critical_questions": ["What should count as a \'sharp fall\': 1%, 3%, 5%, or another threshold?", "What does \'work\' mean here: positive average net return, a high win rate, or outperforming a benchmark?"],\n'
		'  "hypothesis": "A fall of at least 3% in NIFTY 50 produces positive average 5-day forward returns.",\n'
		'  "suggested_experiment": {\n'
		'    "market": "NIFTY 50",\n'
		'    "threshold_percent": 3.0,\n'
		'    "holding_period": 5,\n'
		'    "entry_timing": "next_open",\n'
		'    "exit_price": "close",\n'
		'    "cost_percent": 0.1,\n'
		'    "start_date": "2015-01-01",\n'
		'    "end_date": "2025-12-31",\n'
		'    "max_fall_percent": null,\n'
		'    "min_fall_percent": null\n'
		'  }\n'
		'}\n\n'
		"Detect extra constraints from the question such as fall magnitude limits "
		"(e.g. 'do not consider falls more than 15%' → max_fall_percent = 15.0, "
		"'only consider falls between 3% and 10%' → min_fall_percent = 3.0 and threshold_percent = 10.0). "
		"If a previous_critical_questions list is provided, REMOVE any question that the user has already answered "
		"or that is now resolved by the new question context, and do NOT re-add it.\n"
		f"Question: {question!r}\n"
		f"Current experiment context: {context}.\n"
		f"Previous critical questions: {previous_cq}."
	)
	if not generated:
		return fallback
	try:
		generated["suggested_experiment"] = _validate_experiment(generated["suggested_experiment"])
		if not isinstance(generated.get("interpretation"), str) or not isinstance(generated.get("hypothesis"), str) or not isinstance(generated.get("assumptions"), list) or not isinstance(generated.get("critical_questions"), list):
			raise ValueError
		generated["ai_generated"] = True
		generated["question"] = question
		return generated
	except (KeyError, TypeError, ValueError):
		return fallback


def _explain_results(experiment: dict[str, object], results: dict[str, object]) -> dict[str, object]:
	metrics = results["metrics"]
	trade_count = metrics["trade_count"]
	average = metrics["average_return_pct"] or 0
	win_rate = metrics["win_rate_pct"] or 0
	insufficient_data = trade_count < 25
	fallback = {
		"summary": f"The test found {trade_count} qualifying signals with an average net return of {average:.2f}% and a {win_rate:.1f}% win rate.",
		"conclusion": "The historical sample was positive under these assumptions." if average >= 0 else "The historical sample was negative under these assumptions.",
		"caveats": ["This is a historical sample and does not establish future performance.", "Transaction costs and the selected dates affect the result."],
		"next_questions": ["Does the result survive higher costs?", "Is it consistent across time periods?"],
		"ai_generated": False,
	}
	if insufficient_data:
		fallback["conclusion"] = f"Data is not sufficient: only {trade_count} qualifying signals found (fewer than 25 required for a reliable conclusion)."
		fallback["caveats"] = ["This is a historical sample and does not establish future performance.", "Transaction costs and the selected dates affect the result.", f"Only {trade_count} trades met the signal criteria — too few to draw a statistically meaningful conclusion."]
	generated = _ai_json(
		"Explain these backtest results for a careful, non-promotional research UI. "
		"Return ONLY a JSON object with these exact keys: summary (string), conclusion (string), "
		"caveats (array of strings), next_questions (array of strings).\n\n"
		"Example:\n"
		'{\n'
		'  "summary": "The test found 12 qualifying signals with an average net return of +1.42% and an 58.3% win rate.",\n'
		'  "conclusion": "The historical sample was positive under these assumptions.",\n'
		'  "caveats": ["This is a historical sample and does not establish future performance.", "Transaction costs and the selected dates affect the result."],\n'
		'  "next_questions": ["Does the result survive higher costs?", "Is it consistent across time periods?"]\n'
		'}\n\n'
		"Experiment: "
		+ json.dumps(experiment)
		+ "\nMetrics: "
		+ json.dumps(metrics)
	)
	if generated and all(isinstance(generated.get(key), (str, list)) for key in ("summary", "conclusion", "caveats", "next_questions")):
		generated["ai_generated"] = True
		if insufficient_data:
			generated["conclusion"] = f"Data is not sufficient: only {trade_count} qualifying signals found (fewer than 25 required for a reliable conclusion)."
			generated["caveats"] = ["This is a historical sample and does not establish future performance.", "Transaction costs and the selected dates affect the result.", f"Only {trade_count} trades met the signal criteria — too few to draw a statistically meaningful conclusion."]
		return generated
	return fallback


def _error(message: str, status: int):
	return jsonify({"error": message}), status


@app.post("/api/parse-question")
def parse_question():
	body = request.get_json(silent=True)
	question = body.get("question", "").strip() if isinstance(body, dict) and isinstance(body.get("question", ""), str) else ""
	if not question:
		return _error("question is required", 400)
	if len(question) > 1000:
		return _error("question must be 1000 characters or fewer", 422)
	current_experiment = body.get("current_experiment") if isinstance(body, dict) else None
	if current_experiment is not None:
		try:
			current_experiment = _validate_experiment(current_experiment)
		except RequestError as exc:
			return _error(str(exc), 422)
	previous_critical_questions = body.get("previous_critical_questions") if isinstance(body, dict) else None
	if previous_critical_questions is not None and not isinstance(previous_critical_questions, list):
		return _error("previous_critical_questions must be an array of strings", 422)
	return jsonify(interpret_question(question, current_experiment, previous_critical_questions))


@app.post("/api/backtest")
def backtest_endpoint():
	try:
		experiment = _validate_experiment(request.get_json(silent=True))
		csv_path = Path(DATA_PATHS[experiment["market"]]).expanduser()
		if not csv_path.is_absolute():
			csv_path = Path(__file__).resolve().parent / csv_path
		if not csv_path.is_file():
			return _error(f"historical data is missing for {experiment['market']}: {csv_path}. Run download_data.py for this market or set its CSV path in .env", 503)
		data = load_ohlc_data(csv_path)
		mask = (data["date"] >= experiment["start_date"]) & (data["date"] <= experiment["end_date"])
		data = data.loc[mask].reset_index(drop=True)
		if len(data) < experiment["holding_period"] + 1:
			return _error("the selected date range does not contain enough valid price rows", 422)
		output = run_backtest(
			data,
			threshold_percent=experiment["threshold_percent"],
			holding_period=experiment["holding_period"],
			cost_percent=experiment["cost_percent"],
			max_fall_percent=experiment.get("max_fall_percent"),
			min_fall_percent=experiment.get("min_fall_percent"),
		)
		output["experiment"] = experiment
		output["explanation"] = _explain_results(experiment, output)
		# Benchmark: buy-and-hold return of the index over the test period
		if len(data) >= 2:
			benchmark_return = (data.iloc[-1]["close"] - data.iloc[0]["open"]) / data.iloc[0]["open"] * 100
			output["benchmark_return_pct"] = benchmark_return
		else:
			output["benchmark_return_pct"] = None
		output["price_data"] = [{"date": row.date.strftime("%Y-%m-%d"), "open": float(row.open), "high": float(max(row.open, row.close)), "low": float(min(row.open, row.close)), "close": float(row.close)} for row in data.itertuples()]
		output["signals"] = [{key: trade[key] for key in ("signal_date", "previous_close", "signal_close", "daily_return_pct")} for trade in output["trades"]]
		output["distribution"] = [
			{"label": "Positive", "count": sum(1 for t in output["trades"] if t["net_return_pct"] > 0)},
			{"label": "Negative", "count": len(output["trades"]) - sum(1 for t in output["trades"] if t["net_return_pct"] > 0)},
		]
		output["holding_period_analysis"] = [{"holding_period": experiment["holding_period"], "average_return_pct": output["metrics"]["average_return_pct"] or 0}]
		output["equity_curve"] = []
		return jsonify(output)
	except RequestError as exc:
		return _error(str(exc), 422)
	except (OSError, ValueError, pd.errors.ParserError) as exc:
		return _error(f"could not run backtest: {exc}", 422)


if __name__ == "__main__":
	app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=False)