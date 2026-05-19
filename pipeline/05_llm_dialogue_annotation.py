#!/usr/bin/env python3
"""
LLM-based evaluation of dialogue: dialogue acts, communication quality, linguistic convergence.

Uses OpenAI GPT-4.1-mini for structured JSON evaluation of map task conversations.

Usage:
    from llm_eval import batch_evaluate_session

Requirements:
    pip install openai
"""

import json
import os
from typing import Dict, List, Any, Optional

# ── System prompts ──

MAP_TASK_CONTEXT = """You are an expert analyst evaluating collaborative map task dialogues.

Context: Two participants collaborate on a spatial navigation task. The "Director" sees a map with a route drawn on it and must verbally guide the "Matcher" to reproduce that route on their own (slightly different) map. The Matcher can ask questions but cannot see the Director's map. Each trial lasts ~210 seconds.

The maps contain landmarks (buildings, trees, lakes, etc.) and the route passes near or between these landmarks. Effective communication involves clear spatial references, landmark identification, and turn-by-turn instructions."""


# ── Dialogue Act Classification ──

DIALOGUE_ACT_PROMPT = """Classify each utterance in the following map task dialogue into dialogue act categories.

For each utterance, assign ONE primary dialogue act from this taxonomy:
- INSTRUCT: Giving route directions or spatial instructions
- DESCRIBE: Describing landmarks, map features, or spatial layout
- CHECK: Asking for or giving confirmation ("got it?", "yes", "okay")
- QUERY: Asking a question about the map, route, or instructions
- CLARIFY: Clarifying a previous statement or correcting misunderstanding
- ACKNOWLEDGE: Simple acknowledgment ("mm-hmm", "right", "okay")
- REPAIR: Correcting an error, backtracking, or fixing a mistake
- META: Talk about the task itself, not the map content ("we're running out of time")
- FILLER: Hesitation, false start, or non-content speech ("um", "uh")
- OTHER: Anything that doesn't fit above

Also assign a confidence score (0.0-1.0) for each classification.

Return JSON array with objects: {"utterance_index": int, "text": str, "act": str, "confidence": float}"""


def classify_dialogue_acts(transcript_director: str, transcript_matcher: str,
                           api_key: str = None) -> Dict[str, Any]:
    """Classify dialogue acts for both speakers in a trial."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    dialogue = f"DIRECTOR: {transcript_director}\n\nMATCHER: {transcript_matcher}"

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": MAP_TASK_CONTEXT + "\n\n" + DIALOGUE_ACT_PROMPT},
            {"role": "user", "content": dialogue},
        ],
        temperature=0.1,
        max_tokens=4000,
    )

    try:
        result = json.loads(resp.choices[0].message.content)
    except (json.JSONDecodeError, IndexError):
        result = {"acts": [], "error": "parse_failed"}

    # Compute act distribution
    acts = result.get("acts", result if isinstance(result, list) else [])
    if isinstance(acts, list):
        act_counts = {}
        for a in acts:
            act = a.get("act", "OTHER")
            act_counts[act] = act_counts.get(act, 0) + 1
        total = sum(act_counts.values())
        act_dist = {f"act_pct_{k.lower()}": v / total for k, v in act_counts.items()} if total else {}
    else:
        act_counts = {}
        act_dist = {}

    return {
        "dialogue_acts": acts,
        "act_counts": act_counts,
        **act_dist,
        "total_acts": len(acts) if isinstance(acts, list) else 0,
    }


# ── Communication Quality Rating ──

QUALITY_PROMPT = """Rate the communication quality of this map task dialogue on the following dimensions.
Use a 1-7 Likert scale for each dimension.

Dimensions:
1. **clarity**: How clear and unambiguous are the Director's instructions?
2. **specificity**: How specific are the spatial references? (e.g., "go left" vs "go left past the church about 2cm")
3. **efficiency**: How efficiently is information conveyed? (minimal redundancy, appropriate detail level)
4. **grounding**: How well do participants establish mutual understanding? (checking, confirming, repairing)
5. **adaptiveness**: Does the Director adapt based on Matcher's feedback? Does communication style change?
6. **spatial_precision**: Quality of spatial language (landmarks, directions, distances, relations)
7. **collaboration**: Overall quality of joint effort, turn-taking, and cooperation

Also provide:
- **overall_score**: Single 1-7 overall communication quality rating
- **strengths**: List of 1-3 communication strengths (brief)
- **weaknesses**: List of 1-3 communication weaknesses (brief)

Return JSON with all ratings and text fields."""


def rate_communication_quality(transcript_director: str, transcript_matcher: str,
                               api_key: str = None) -> Dict[str, Any]:
    """Rate communication quality on multiple dimensions."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    dialogue = f"DIRECTOR: {transcript_director}\n\nMATCHER: {transcript_matcher}"

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": MAP_TASK_CONTEXT + "\n\n" + QUALITY_PROMPT},
            {"role": "user", "content": dialogue},
        ],
        temperature=0.2,
        max_tokens=2000,
    )

    try:
        result = json.loads(resp.choices[0].message.content)
    except (json.JSONDecodeError, IndexError):
        result = {"error": "parse_failed"}

    # Extract numeric ratings
    ratings = {}
    for dim in ["clarity", "specificity", "efficiency", "grounding",
                 "adaptiveness", "spatial_precision", "collaboration", "overall_score"]:
        val = result.get(dim)
        if isinstance(val, (int, float)):
            ratings[f"quality_{dim}"] = float(val)
        elif isinstance(val, dict) and "score" in val:
            ratings[f"quality_{dim}"] = float(val["score"])

    ratings["quality_strengths"] = result.get("strengths", [])
    ratings["quality_weaknesses"] = result.get("weaknesses", [])

    return ratings


# ── Linguistic Convergence Assessment ──

CONVERGENCE_PROMPT = """Analyze linguistic convergence/alignment between the Director and Matcher in this map task dialogue.

Assess:
1. **lexical_convergence** (1-7): Do they adopt each other's vocabulary? Do they converge on shared terms for landmarks and directions?
2. **syntactic_convergence** (1-7): Do they mirror each other's sentence structures?
3. **spatial_frame_alignment** (1-7): Do they use the same spatial reference frame? (e.g., both using "left/right" vs one using "north/south")
4. **conceptual_convergence** (1-7): Do they develop shared mental models of the map? Do they build on each other's descriptions?
5. **landmark_agreement** (1-7): Do they consistently use the same names/descriptions for landmarks?

Also provide:
- **convergence_trajectory**: "increasing", "decreasing", "stable", or "mixed" — does alignment improve over the conversation?
- **shared_vocabulary**: list of key terms both speakers use for the same referents
- **misalignments**: list of terms or concepts where speakers diverge

Return JSON with all fields."""


def assess_linguistic_convergence(transcript_director: str, transcript_matcher: str,
                                  api_key: str = None) -> Dict[str, Any]:
    """Assess linguistic convergence between dyad members."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    dialogue = f"DIRECTOR: {transcript_director}\n\nMATCHER: {transcript_matcher}"

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": MAP_TASK_CONTEXT + "\n\n" + CONVERGENCE_PROMPT},
            {"role": "user", "content": dialogue},
        ],
        temperature=0.2,
        max_tokens=2000,
    )

    try:
        result = json.loads(resp.choices[0].message.content)
    except (json.JSONDecodeError, IndexError):
        result = {"error": "parse_failed"}

    feats = {}
    for dim in ["lexical_convergence", "syntactic_convergence", "spatial_frame_alignment",
                 "conceptual_convergence", "landmark_agreement"]:
        val = result.get(dim)
        if isinstance(val, (int, float)):
            feats[f"conv_{dim}"] = float(val)

    feats["conv_trajectory"] = result.get("convergence_trajectory", "")
    feats["conv_shared_vocabulary"] = result.get("shared_vocabulary", [])
    feats["conv_misalignments"] = result.get("misalignments", [])

    # Compute mean convergence score
    scores = [v for k, v in feats.items() if k.startswith("conv_") and isinstance(v, (int, float))]
    feats["conv_mean"] = float(sum(scores) / len(scores)) if scores else 0.0

    return feats


# ── Batch evaluation ──

def evaluate_trial(transcript_director: str, transcript_matcher: str,
                   trial: int, api_key: str = None) -> Dict[str, Any]:
    """Run all LLM evaluations for a single trial."""
    result = {"trial": trial}

    if not transcript_director.strip() and not transcript_matcher.strip():
        result["llm_eval_error"] = "empty_transcripts"
        return result

    # Dialogue acts
    try:
        acts = classify_dialogue_acts(transcript_director, transcript_matcher, api_key)
        # Flatten numeric features (skip nested lists)
        for k, v in acts.items():
            if isinstance(v, (int, float, str)):
                result[k] = v
            elif k.startswith("act_pct_"):
                result[k] = v
    except Exception as e:
        result["dialogue_act_error"] = str(e)[:200]

    # Communication quality
    try:
        quality = rate_communication_quality(transcript_director, transcript_matcher, api_key)
        for k, v in quality.items():
            if isinstance(v, (int, float, str)):
                result[k] = v
    except Exception as e:
        result["quality_error"] = str(e)[:200]

    # Linguistic convergence
    try:
        conv = assess_linguistic_convergence(transcript_director, transcript_matcher, api_key)
        for k, v in conv.items():
            if isinstance(v, (int, float, str)):
                result[k] = v
    except Exception as e:
        result["convergence_error"] = str(e)[:200]

    return result


def batch_evaluate_session(transcripts: List[Dict[str, str]],
                           api_key: str = None) -> List[Dict[str, Any]]:
    """
    Evaluate all trials in a session.

    Args:
        transcripts: list of dicts with keys:
            - trial (int)
            - director_text (str)
            - matcher_text (str)
        api_key: OpenAI API key

    Returns:
        List of evaluation dicts, one per trial.
    """
    results = []
    for t in transcripts:
        trial = t.get("trial", 0)
        d_text = t.get("director_text", "")
        m_text = t.get("matcher_text", "")
        result = evaluate_trial(d_text, m_text, trial, api_key)
        result["sessionId"] = t.get("sessionId", "")
        results.append(result)
    return results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="LLM evaluation of map task dialogue")
    ap.add_argument("--director-text", required=True, help="Director transcript text")
    ap.add_argument("--matcher-text", required=True, help="Matcher transcript text")
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--api-key", default=None)
    args = ap.parse_args()

    result = evaluate_trial(args.director_text, args.matcher_text, args.trial, args.api_key)
    print(json.dumps(result, indent=2, default=str))
