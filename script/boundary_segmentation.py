import argparse
import json
import os
import re
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Set

from tqdm import tqdm
import requests


# DeepSeek API endpoint for chat completions
API_URL = "https://api.deepseek.com/v1/chat/completions"


@dataclass
class PairDecision:
    left_index: int
    right_index: int
    decision: str
    confidence: float
    reason: str


@dataclass
class MergeStep:
    step_id: int
    paragraph_indices: List[int]
    cognitive_mode: str
    subgoal: str
    summary: str
    outcome: str
    text: str


@dataclass(frozen=True)
class SegmentationProfile:
    name: str
    system_boundary: str
    system_rename: str
    boundary_extra_rules: Tuple[str, ...] = ()
    rename_extra_rules: Tuple[str, ...] = ()
    rename_history_limit: Optional[int] = None
    duplicate_subgoal_similarity: float = 0.9
    duplicate_outcome_similarity: float = 0.85
    casework_merge: bool = False
    casework_subgoal_similarity: float = 0.9
    exploration_merge: bool = False
    exploration_subgoal_similarity: float = 0.7
    model_search_merge: bool = False
    model_search_subgoal_similarity: float = 0.45
    aggressive_refine_merge: bool = False
    refine_subgoal_similarity: float = 0.85
    align_groups_to_steps: bool = False
    inline_error_result: bool = False
    preserve_error_schema: bool = False


SYSTEM_BOUNDARY = (
    "You are a reasoning-trajectory segmenter. "
    "Decide whether the right paragraph should start a NEW training step. "
    "Return JSON only."
)

SYSTEM_RENAME = (
    "You are a high-level planner labeler. "
    "Given a merged reasoning chunk, produce one dominant cognitive mode, one concise subgoal name, a rich summary, and a short outcome. "
    "Use neutral factual style without personal pronouns. "
    "Return JSON only."
)

COGNITIVE_MODE_CANDIDATES = [
    "ProblemUnderstanding",
    "Decomposition",
    "Exploration",
    "DeepReasoning",
    "Calculation",
    "Verification",
    "Reflection",
    "Backtracking",
    "Synthesis",
    "Finalization",
    "Other",
]

MERGEABLE_FOLLOWUP_MODES = {"Finalization"}
REFINEMENT_CUES = [
    "correct",
    "correction",
    "incorrect",
    "mistake",
    "fix",
    "revise",
    "revised",
    "previous error",
    "earlier error",
    "instead",
    "not valid",
    "wrong",
    "misread",
    "recompute",
]

METHOD_DISTINCTION_CUES = [
    "alternative method",
    "alternative route",
    "another way",
    "different method",
    "directly",
    "by substitution",
    "via substitution",
    "via slope",
    "via formula",
    "via shoelace",
    "via vector",
    "graph",
    "table",
]

PACKAGING_CUES = [
    "boxed",
    "box",
    "final answer",
    "format",
    "notation",
    "representation",
    "fraction",
    "decimal",
    "exact form",
    "simplified form",
    "interval notation",
]

FINAL_ANSWER_CUES = [
    "the answer is",
    "answer should be",
    "thus the answer",
    "therefore the answer",
    "hence the answer",
    "so the answer",
    "final result",
    "final value",
    "same answer",
    "same result",
    "yields the same result",
]

CALCULATION_REVISIT_CUES = [
    "again",
    "recompute",
    "recalculated",
    "recalculate",
    "confirm",
    "checked",
    "for verification",
    "for consistency",
]

LOCAL_CHECK_CUES = [
    "component",
    "term",
    "partial sum",
    "order of operations",
    "arithmetic",
    "multiply",
    "multiplied",
    "addition",
    "subtraction",
    "rechecked",
    "same value",
    "same quantity",
    "plugged in the same values",
]

CONTINUATION_CUES = [
    "continue",
    "continued",
    "continuing",
    "complete",
    "completed",
    "completing",
    "remaining",
    "rest of",
    "partial",
    "running",
    "stepwise",
    "lower bound",
    "upper bound",
    "next term",
    "final term",
]

INITIATION_CUES = [
    "initiated",
    "began",
    "started",
    "set up",
    "identified initial",
    "initial terms",
    "ready for",
    "framework for",
]

MICRO_CALCULATION_CUES = [
    "inequality",
    "bound",
    "cube",
    "term",
    "terms",
    "coefficient",
    "coefficients",
    "expansion",
    "product",
    "partial product",
    "partial sum",
    "summation",
    "multiplication",
    "subtraction",
    "difference",
    "divide",
    "division",
    "power",
    "squared",
    "cubed",
]

FOCUS_TOKEN_STOPWORDS = {
    "a",
    "an",
    "and",
    "answer",
    "apply",
    "as",
    "by",
    "check",
    "combine",
    "complete",
    "compute",
    "confirm",
    "continue",
    "derive",
    "determine",
    "equation",
    "evaluate",
    "expression",
    "find",
    "final",
    "for",
    "form",
    "from",
    "identify",
    "into",
    "is",
    "list",
    "obtain",
    "of",
    "partial",
    "remaining",
    "result",
    "set",
    "simplify",
    "solve",
    "sum",
    "the",
    "to",
    "up",
    "use",
    "via",
    "verify",
    "with",
}

CASEWORK_VERBS = (
    "test",
    "verify",
    "check",
    "consider",
    "evaluate",
    "reject",
    "confirm",
    "reconfirm",
    "compute",
)

CASEWORK_CUES = (
    "candidate",
    "candidates",
    "score",
    "scores",
    "value",
    "values",
    "case",
    "cases",
    "representability",
    "ambiguity",
    "unique",
    "uniqueness",
    "possibility",
)

EXPLORATION_VERBS = (
    "consider",
    "explore",
    "analyze",
    "visualize",
    "hypothesize",
    "assess",
    "reassess",
    "investigate",
    "examine",
    "interpret",
    "identify",
    "select",
    "compare",
    "plan",
)

MODEL_SEARCH_CUES = (
    "model",
    "geometry",
    "geometric",
    "folding",
    "folded",
    "net",
    "solid",
    "polyhedron",
    "polyhedral",
    "volume",
    "decomposition",
    "decompose",
    "tetrahedron",
    "tetrahedral",
    "pyramid",
    "prism",
    "prismatoid",
    "octahedron",
    "coordinate",
    "coordinates",
    "cross-sectional",
    "cross section",
    "face",
    "faces",
    "vertex",
    "vertices",
    "edge",
    "edges",
    "apex",
)

PROFILE_ALIASES = {
    "math": "math",
    "default": "math",
    "generic": "math",
    "aime": "aime",
    "aime24": "aime",
    "gpqa": "gpqa",
    "gpqaextended": "gpqa",
    "boardgameqa": "boardgameqa",
    "boardgame": "boardgameqa",
    "boardgameqav1": "boardgameqa",
}

PROFILES = {
    "math": SegmentationProfile(
        name="math",
        system_boundary=SYSTEM_BOUNDARY,
        system_rename=SYSTEM_RENAME,
    ),
    "aime": SegmentationProfile(
        name="aime",
        system_boundary=(
            "You are a reasoning-trajectory segmenter for short-form contest math. "
            "Prefer phase-level reusable steps over microscopic case-by-case fragmentation. "
            "Keep setup, sustained case sweeps, decisive verification, and final answer packaging as a small number of stable steps whenever possible. "
            "Decide whether the right paragraph should start a NEW training step. "
            "Return JSON only."
        ),
        system_rename=(
            "You are a high-level planner labeler for contest-math reasoning. "
            "Compress repetitive candidate testing, repetitive setup, and short local checks into reusable phase-level step names, "
            "avoid nested Refine chains, and prefer stable step identities over per-number labels. "
            "Treat the final answer plus a brief final check as one finalization step whenever possible. "
            "Use neutral factual style without personal pronouns. "
            "Return JSON only."
        ),
        boundary_extra_rules=(
            "YES when adjacent paragraphs are still unpacking givens, introducing variables, recalling one formula family, or setting up one shared equation system.",
            "YES when adjacent paragraphs continue the same exhaustive search, candidate sweep, or repeated testing pattern over nearby values or cases.",
            "YES when the right paragraph only tests another candidate under the same criterion, table, or complement check.",
            "YES when adjacent paragraphs continue the same systematic verification sweep, small-case sanity check, or recomputation of the same count or threshold under one unchanged counting principle.",
            "YES when adjacent paragraphs continue exploring nearby geometric interpretations, nearby solid-model hypotheses, or nearby algebraic routes without establishing a new derived deliverable.",
            "YES when adjacent paragraphs continue trying nearby solid models, volume formulas, coordinate decompositions, or folding interpretations for the same figure; keep that hypothesis search as one reusable step unless a decisive model is actually selected.",
            "YES when the right paragraph only restates the same candidate answer, performs a brief consistency check, or boxes an already established result.",
            "NO only when the search changes criterion, changes representation, switches method, or promotes the tested cases into a new conclusion.",
        ),
        rename_extra_rules=(
            "For AIME-style solutions, prefer phase-level labels such as setup, derive relation, sweep cases, verify decisive candidate, and finalize answer.",
            "Compress repeated setup into one reusable subgoal instead of separate labels for restating givens, naming variables, and recalling formulas.",
            "For AIME-style solutions, compress repeated candidate checks under one reusable subgoal instead of naming one step per tested number.",
            "Prefer broad labels such as 'Test candidate values for representability' over labels such as 'Test 38 for representability' unless one number is the decisive turning point.",
            "When several adjacent chunks explore nearby geometric models or nearby candidate structures, prefer one stable exploration label such as 'Explore candidate solid models' instead of one label per model.",
            "When several adjacent chunks test nearby scores, nearby values, or nearby cases under the same ambiguity criterion, prefer one stable verification label rather than one label per score or value.",
            "For long case sweeps, use a fresh subgoal only when the reasoning switches criterion, starts a different route, or isolates the decisive candidate.",
            "When several adjacent chunks only verify the same counting formula, threshold, or candidate answer on nearby small cases, keep them under one stable verification label instead of creating one step per example.",
            "When a chunk only recomputes the same probability, sum, threshold, or representability count by a nearby arithmetic variant, merge it into the same verification or calculation phase unless it changes the decisive criterion.",
            "For geometry-heavy AIME solutions, compress repeated solid-model testing, volume-formula trials, and coordinate-decomposition attempts into one stable exploration or decomposition label instead of creating one step per candidate model.",
            "If several adjacent chunks only reject nearby polyhedron, prism, pyramid, octahedron, or coordinate models for the same figure, keep them inside one model-search step until a decisive construction is selected.",
            "If the final chunk states the answer and only briefly rechecks it, label it Finalization rather than Synthesis or Verification.",
            "Avoid nested 'Refine:' chains. If the chunk is not a true correction, give it a fresh concrete subgoal name.",
            "If the chunk continues the same case sweep, complement check, or candidate table, keep the same local deliverable rather than introducing a new subgoal.",
        ),
        rename_history_limit=6,
        duplicate_subgoal_similarity=0.8,
        duplicate_outcome_similarity=0.7,
        casework_merge=True,
        casework_subgoal_similarity=0.72,
        exploration_merge=True,
        exploration_subgoal_similarity=0.5,
        model_search_merge=True,
        model_search_subgoal_similarity=0.38,
        aggressive_refine_merge=True,
        refine_subgoal_similarity=0.74,
        inline_error_result=True,
        preserve_error_schema=True,
    ),
    "gpqa": SegmentationProfile(
        name="gpqa",
        system_boundary=(
            "You are a reasoning-trajectory segmenter for expert-level multiple-choice science QA. "
            "Prefer criterion-level or option-family-level steps over microscopic sentence-by-sentence fragmentation. "
            "Aggressively avoid single-sentence fragments when they only restate an option, quote a formula, state a numeric match, name a surviving choice, or perform a brief elimination under an unchanged scientific criterion. "
            "Keep consecutive checks under the same evidence source, contradiction pattern, or physical principle inside one reusable step whenever possible. "
            "Decide whether the right paragraph should start a NEW training step. "
            "Return JSON only."
        ),
        system_rename=(
            "You are a high-level planner labeler for expert multiple-choice science reasoning. "
            "Prefer stable criterion-level option-evaluation or mechanism-evaluation step names, compress repetitive elimination under one reusable label, "
            "and avoid labels that merely echo a short fragment, an answer letter, a copied formula line, or a numeric answer candidate. "
            "Use compact reusable labels, typically 4 to 9 words, that name the scientific decision rather than the surface text. "
            "and only create a new step when the scientific criterion or local deliverable truly changes. "
            "Use neutral factual style without personal pronouns. "
            "Return JSON only."
        ),
        boundary_extra_rules=(
            "YES when adjacent paragraphs continue evaluating the same answer option, scientific mechanism, or elimination criterion.",
            "YES when the right paragraph adds supporting or disconfirming evidence for the same candidate answer or compares nearby options under the same principle.",
            "YES when the right paragraph only restates an option letter, copies a formula, names a candidate, or gives a one-line elimination while the scientific criterion is unchanged.",
            "YES when the right paragraph only says a candidate survives, a candidate fails, or a final option is selected after the previous chunk already established the deciding criterion.",
            "YES when adjacent paragraphs are checking multiple answer choices under the same scientific test, contradiction, or evidence source; keep that as one reusable step whenever possible.",
            "YES when the right paragraph is only a brief final-answer sentence or light confidence check attached to an already established conclusion.",
            "YES when the right paragraph only states a formula, a spectral peak, a sequence fragment, or a numeric intermediate that is immediately used by the same ongoing option-analysis step.",
            "YES when adjacent paragraphs keep screening options against the same table, spectrum, expression pattern, conservation law, or mechanistic criterion.",
            "NO only when the reasoning switches to a materially different scientific criterion, a different option family, or a final cross-option synthesis.",
        ),
        rename_extra_rules=(
            "For GPQA-style reasoning, prefer criterion-level or option-family-level subgoals over sentence-level paraphrases.",
            "Compress repeated elimination or validation of nearby answer options under one stable subgoal when they rely on the same scientific principle.",
            "Prefer labels such as 'Eliminate options C and D via symmetry argument' or 'Evaluate option B against expression data' over labels that just repeat short text fragments.",
            "Do not use answer letters, copied equations, or ultra-local phrases as the whole subgoal unless they are necessary to disambiguate between options.",
            "If the chunk only contains a formula line, a numeric value, a sequence fragment, or a peak assignment, name the scientific use of that evidence rather than the literal text.",
            "If a chunk is extremely short and only states a surviving option, a duplicated choice, or a final choice letter, name the underlying scientific decision instead of the literal text.",
            "When the chunk moves from screening options to selecting the best answer, use a fresh synthesis or finalization label instead of repeating the prior option-analysis subgoal.",
            "Prefer one finalization step that states the final answer and any brief final confirmation, rather than splitting answer selection into multiple micro-steps.",
            "Prefer 4 to 9 words for the subgoal; use 10 to 12 only when needed for scientific specificity.",
            "Avoid creating separate subgoals for consecutive option checks if one criterion-level label can cover the whole mini-sequence.",
        ),
        rename_history_limit=6,
        duplicate_subgoal_similarity=0.8,
        duplicate_outcome_similarity=0.74,
        casework_merge=True,
        casework_subgoal_similarity=0.82,
        align_groups_to_steps=True,
        inline_error_result=True,
    ),
    "boardgameqa": SegmentationProfile(
        name="boardgameqa",
        system_boundary=(
            "You are a reasoning-trajectory segmenter for rule-based BoardgameQA reasoning. "
            "Prefer phase-level steps such as setup, rule application, conflict resolution, and answer finalization over microscopic sentence-by-sentence fragmentation. "
            "Keep all raw game-state extraction, rule quoting, and question restatement before the first real inference inside one setup step whenever possible. "
            "Do not keep the whole trajectory as one step once the reasoning moves from setup into actually applying a rule or resolving a preference conflict. "
            "For simple one-rule problems, the default target is 2 or 3 steps total: setup, apply the active rule, and optional short finalization. Use 4 or more steps only if the text explicitly develops a second active rule, an explicit preference comparison, or a genuinely separate ambiguity analysis that changes what must be decided. "
            "For one-rule samples with no explicit preference relation, once the rule has been applied and the answer is known, every later sentence about no conflicts, consistency, hidden contradictions, trick interpretations, or final yes/no wording must stay inside that same final application/finalization step. "
            "After a rule has already been applied, do not start a new step just because the text restates the same consequence, notes that no other preferences are given, double-checks the same premise, resolves a small wording or pronoun ambiguity, considers a minor alternative interpretation, or rewrites the answer in yes/no form. "
            "Do not create multiple post-application verification steps unless a genuinely new competing rule, explicit preference ordering, or real contradiction analysis appears. "
            "If there is only one active rule in the sample, assume later confirmation sentences belong to that same rule-application phase unless a genuinely new deliverable appears. "
            "If the sample contains only one explicit rule and no explicit preference relation, prefer merging the final answer sentence and any brief 'no contradiction' note into the same application/finalization step instead of splitting them apart. "
            "Do not create a separate step for pronoun resolution, wording equivalence, entity matching, hidden-contradiction checks, trick-interpretation checks, or 'no hidden contradiction' discussion when those only support the same already-applied rule. "
            "Once a one-rule sample has derived the answer, keep every later sentence that merely reconfirms the same answer inside that same final application/finalization step. "
            "Decide whether the right paragraph should start a NEW training step. "
            "Return JSON only."
        ),
        system_rename=(
            "You are a high-level planner labeler for rule-based BoardgameQA reasoning. "
            "Prefer stable phase-level steps such as extracting the relevant state and active rules, applying one rule, resolving rule conflicts by preference, "
            "and finalizing the answer, while avoiding labels for tiny restatements, leftover bullets, or isolated irrelevant-fact notes. "
            "Do not label a chunk as ProblemUnderstanding if it already derives that a rule applies, fails, or determines the final yes/no outcome. "
            "For simple one-rule examples, prefer 2 or 3 stable steps rather than chains of separate application, verification, reconfirmation, reflection, and boxed-answer steps. "
            "If a one-rule sample already derived the answer, do not create a second post-application step just for hidden-contradiction checks, trick-interpretation checks, brief reconfirmation, or restating the same yes/no answer. "
            "If a one-rule sample would otherwise produce labels like 'Check for conflicting rules or preferences', 'Reconfirm rule applicability and answer consistency', 'Synthesize final yes/no answer', or 'State final answer and confirm no competing rules', collapse those chunks into the same final application/finalization step instead. "
            "Do not create a separate Verification, Reflection, or Finalization step when the chunk only repeats an already established rule conclusion, briefly notes that no competing rules are present, resolves a small wording/pronoun match inside the same rule, considers a minor interpretation issue that does not change the answer, or formats the same yes/no answer. "
            "Use at most one Finalization step per sample unless there is a real earlier false start that later gets corrected. "
            "For one-rule samples, avoid more than one post-application step unless a real contradiction or preference comparison appears. "
            "If a one-rule sample has already derived the answer, prefer folding the final answer sentence and short 'no conflict' commentary into that same step rather than creating a separate finalization step. "
            "If no explicit second rule or preference relation appears, avoid separate subgoals for conflict-checking, pronoun resolution, wording equivalence, or answer packaging. "
            "Never put raw quotation marks from the reasoning trace inside JSON string values; paraphrase quoted rule wording instead. "
            "Use neutral factual style without personal pronouns. "
            "Return JSON only."
        ),
        boundary_extra_rules=(
            "YES when adjacent paragraphs are still collecting raw game-state facts, quoting rules, restating the question, or listing preferences before any actual inference starts.",
            "YES when adjacent paragraphs continue checking the same rule, the same antecedent conditions, or the same game-state evidence.",
            "YES when the right paragraph only filters irrelevant facts, restates the same rule consequence, or continues the same preference/conflict analysis.",
            "YES when the right paragraph only rechecks the same satisfied rule condition, repeats that no other rules or preferences are given, or rewrites the already established conclusion in yes/no language.",
            "YES when the right paragraph only resolves a small wording match, entity reference, or pronoun reference inside the same already-applied rule.",
            "YES when the right paragraph only raises and dismisses a minor alternative interpretation, hidden contradiction, or trick reading without changing which rule applies or what answer follows.",
            "YES when the right paragraph only states the boxed answer or a brief final confirmation after the conclusion is already established.",
            "YES when the right paragraph only says that no competing rules, no preferences, or no contradictions change the already established one-rule conclusion.",
            "YES when the right paragraph only confirms wording equivalence, entity identity, or pronoun reference needed for the same already-applied rule.",
            "YES when the right paragraph only repeats the final yes/no answer in slightly different wording after the prior chunk already established it.",
            "YES when a one-rule sample's right paragraph would only motivate labels like conflict-check, reconfirmation, hidden-contradiction check, or final yes/no restatement after the previous chunk already derived the same answer.",
            "NO when the reasoning moves from setup into the first real inference, such as deciding whether a rule condition is satisfied, whether a rule applies, or whether a conclusion follows.",
            "NO when the reasoning explicitly shifts from one active rule to a competing rule, from rule application to preference/conflict resolution, or to a genuinely new synthesis that combines distinct earlier results.",
            "NO only if the right paragraph introduces a real second line of reasoning; if it merely reconfirms the same applied rule, answer YES.",
        ),
        rename_extra_rules=(
            "For BoardgameQA-style reasoning, prefer phase-level subgoals such as 'Extract relevant state and active rules', 'Apply Rule 2 to derive no-roll conclusion', 'Resolve Rule 1 versus Rule 2 by preference', or 'Finalize yes/no answer'.",
            "Treat all pre-inference setup material as one setup step whenever it is mainly listing facts, rules, preferences, or the question.",
            "Do not split off a separate step just because one leftover bullet, one repeated rule quote, or one short irrelevant-fact note appears inside the same setup or rule-application phase.",
            "When the chunk first derives that a rule applies, fails, or determines the answer, give it an application or synthesis label rather than ProblemUnderstanding.",
            "If the conclusion already follows from one active rule, keep later restatement, brief double-checking, no-other-rules comments, and yes/no phrasing inside the same application or finalization step instead of creating extra verification steps.",
            "Do not create a dedicated conflict-check step unless the chunk actually mentions a second rule, an explicit preference ordering, or a real contradiction to resolve.",
            "Do not create separate steps for pronoun resolution, wording equivalence, or entity-name matching if they only support the same already-active rule.",
            "Do not use Reflection for a minor interpretive aside unless that aside genuinely changes the reasoning route or forces a revised answer.",
            "When the chunk moves from checking rule applicability to resolving a conflict or preference ordering, use a fresh subgoal for that transition.",
            "Keep answer-formatting, brief restatement, and boxed-answer packaging inside the same finalization step rather than splitting them apart.",
            "For simple single-rule examples, prefer 2 or 3 steps total: setup, rule application, and optional short finalization unless the trajectory truly contains a real competing-rule analysis.",
            "For a one-rule sample with no explicit preference relation, do not introduce a separate 'check conflicts' or separate finalization subgoal unless the text actually performs a new contradiction analysis.",
            "If a later chunk only checks for hidden contradictions, trick interpretations, or repeated yes/no phrasing after the answer is already known, merge it into the preceding application or finalization step.",
            "If a later chunk only says the same answer still holds, merge it into the preceding application or finalization step instead of naming a new verification step.",
            "If adjacent one-rule chunks would naturally receive labels such as 'Check for conflicting rules or preferences', 'Reconfirm rule applicability and answer consistency', 'Synthesize final yes answer', or 'State final answer and confirm no competing rules', merge them into one final application/finalization step instead of keeping multiple tail steps.",
            "When emitting JSON-valued text fields, paraphrase any quoted rule wording and avoid literal double quotes inside the field text.",
        ),
        rename_history_limit=2,
        duplicate_subgoal_similarity=0.72,
        duplicate_outcome_similarity=0.66,
        casework_merge=True,
        casework_subgoal_similarity=0.82,
        align_groups_to_steps=True,
        inline_error_result=True,
        preserve_error_schema=True,
    ),
}


def normalize_profile_key(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    lowered = re.sub(r"[^a-z0-9]+", "", value.lower())
    if lowered.startswith("boardgameqa") or lowered.startswith("boardgame"):
        return "boardgameqa"
    return PROFILE_ALIASES.get(lowered)


def resolve_profile(profile_name: str, dataset_name: Optional[str], input_jsonl: str) -> SegmentationProfile:
    if profile_name != "auto":
        resolved = normalize_profile_key(profile_name)
        if resolved is None:
            raise ValueError(f"Unknown profile: {profile_name}")
        return PROFILES[resolved]

    input_stem = os.path.splitext(os.path.basename(input_jsonl))[0]
    input_parent = os.path.basename(os.path.dirname(input_jsonl))
    for candidate in (dataset_name, input_stem, input_parent):
        resolved = normalize_profile_key(candidate)
        if resolved is not None:
            return PROFILES[resolved]

    if "boardgame" in input_jsonl.lower():
        return PROFILES["boardgameqa"]

    if "aime" in input_jsonl.lower():
        return PROFILES["aime"]

    return PROFILES["math"]


def render_rule_block(lines: Tuple[str, ...]) -> str:
    if not lines:
        return ""
    return "\n".join(f"- {line}" for line in lines)


def strip_refine_prefix(text: str) -> str:
    return re.sub(r"^(?:refine:\s*)+", "", text.strip(), flags=re.IGNORECASE)


def make_refine_subgoal(text: str) -> str:
    base = strip_refine_prefix(text)
    return f"Refine: {base}" if base else "Refine"


def call_deepseek(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    timeout: int = 120,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    resp = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def repair_common_invalid_escapes(s: str) -> str:
    # Convert invalid JSON escapes into a valid format:
    # \d -> \\d, \( -> \\(, \[ -> \\[
    # But keep valid escapes: \", \\, \/, \b, \f, \n, \r, \t, \u
    return re.sub(r'\\([^"\\/bfnrtu])', r'\\\\\1', s)


def extract_json(text: str) -> Dict:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    candidates = [text]

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        candidate = match.group(0)
        if candidate != text:
            candidates.append(candidate)

    last_err = None
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            last_err = e

        try:
            repaired = repair_common_invalid_escapes(candidate)
            return json.loads(repaired)
        except json.JSONDecodeError as e:
            last_err = e

    raise last_err


def split_paragraphs(cot_text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", cot_text.strip()) if p.strip()]


def auto_merge_paragraphs(
    paragraphs: List[str],
    auto_merge_threshold: Optional[int],
    auto_merge_group_size: Optional[int],
) -> List[str]:
    if auto_merge_threshold is None or auto_merge_group_size is None:
        return paragraphs
    if auto_merge_threshold <= 0 or auto_merge_group_size <= 1:
        return paragraphs
    if len(paragraphs) <= auto_merge_threshold:
        return paragraphs

    merged = []
    for start in range(0, len(paragraphs), auto_merge_group_size):
        chunk = paragraphs[start : start + auto_merge_group_size]
        merged.append("\n\n".join(chunk))
    return merged


def load_env_file(env_path: str) -> None:
    if not env_path or not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            row = line.strip()
            if not row or row.startswith("#") or "=" not in row:
                continue
            key, value = row.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def boundary_prompt(profile: SegmentationProfile, problem: str, left: str, right: str) -> str:
    extra_rules = render_rule_block(profile.boundary_extra_rules)
    extra_section = f"\n{extra_rules}" if extra_rules else ""

    return f"""Task: Decide whether the right paragraph should start a NEW training step.

Goal:
- Prefer reusable reasoning steps, but avoid fragmenting one local deliverable into multiple tiny follow-up steps.

Decision policy:
- YES if both paragraphs are still working toward the same local deliverable.
- YES when the right paragraph is a short continuation, brief verification, restatement, answer-formatting step, representation choice, or light elaboration of the same result.
- YES when the right paragraph recomputes or confirms the same quantity without changing the route's main deliverable.
- YES when the right paragraph only rechecks a local arithmetic sub-result, a single term, or one component of the same ongoing calculation without introducing a genuinely new method.
- YES when the right paragraph continues the same multi-line calculation, such as finishing an inequality bound, listing remaining expansion terms, completing a coefficient sum, continuing a power/product computation, or filling in partial products from the same setup.
- YES when the left paragraph starts a computation and the right paragraph simply carries out the remaining arithmetic details of that same computation.
- YES when the right paragraph only chooses between equivalent answer forms, such as decimal versus fraction, exact form versus approximate form, or plain answer versus boxed notation.
- NO if the right paragraph launches a substantial alternative method or an independent verification route that could stand alone as its own training step, even when it reaches the same answer.
- NO only if the right paragraph clearly introduces a new local deliverable: a new case split, a new derivation target, solving a different unknown, switching to a substantially different method, combining earlier results into a new conclusion, or presenting a materially new answer that was not already established.
- Self-correction without changing the local deliverable => YES.
- Judge by change of local deliverable, not by paragraph break, writing continuity, shared symbols, or topical overlap.
- If uncertain and no new deliverable is explicit, answer YES.{extra_section}

Problem:
{problem}

[Left paragraph]
{left}

[Right paragraph]
{right}

Output JSON schema:
{{
    "left_objective": "one short phrase",
    "right_objective": "one short phrase",
  "decision": "YES" or "NO",
    "change_type": "same_objective | new_case | new_derivation_target | calculation_to_verification | verification_to_finalization | backtrack | synthesis | other",
  "confidence": 0.0 to 1.0,
  "reason": "one sentence"
}}
"""


def rename_prompt(
    profile: SegmentationProfile,
    problem: str,
    merged_text: str,
    previous_steps: List[MergeStep],
    is_last_chunk: bool,
) -> str:
    history_steps = previous_steps
    history_note = "Use all prior steps to avoid repeated subgoal names."
    if profile.rename_history_limit is not None and len(previous_steps) > profile.rename_history_limit:
        history_steps = previous_steps[-profile.rename_history_limit :]
        history_note = (
            f"Only the most recent {len(history_steps)} prior steps are shown once the trajectory becomes long. "
            "Use them to avoid local repetition, but do not overfit to old wording."
        )

    if history_steps:
        history_text = "\n".join(
            f"{step.step_id}. {step.cognitive_mode} | {step.subgoal} | {step.summary}" for step in history_steps
        )
    else:
        history_text = "None"

    extra_rules = render_rule_block(profile.rename_extra_rules)
    extra_section = extra_rules if extra_rules else "- Use one stable reusable subgoal name for the chunk."
    last_chunk_note = (
        "- This is the final chunk of the trajectory. If it mainly states, packages, or briefly confirms the final answer, label it Finalization."
        if is_last_chunk
        else "- This is not the final chunk of the trajectory. Avoid using Finalization unless the chunk is clearly only packaging the final answer."
    )

    return f"""Task: name this merged chunk as ONE subgoal.

Constraints:
- English.
- action + object style.
- <= 12 words.
- Do not include final answer formatting.
- Avoid repeating prior subgoal names.
- Use Refine: <prior subgoal> only for a genuine correction or repair of an earlier incorrect or incomplete step.
- Do not use Refine for routine verification, restatement, boxed-answer formatting, or alternative derivations that reach the same result.
- The subgoal must distinguish this chunk from adjacent chunks in the same sample.
- If two chunks verify the same result using different methods, keep both steps but distinguish the subgoal names by method, such as "Verify g(3) via inverse evaluation" versus "Verify g(3) via direct substitution".
- Reusing an earlier subgoal name is allowed only for a true continuation of the same local computation, not for a distinct verification route.
- If a chunk only chooses answer representation, notation, or boxed formatting for an already established result, keep it within the same finalization step rather than creating multiple finalization steps.
- If a chunk states a candidate final answer but the reasoning later continues with substantial new derivation or verification, do not label that earlier chunk as Finalization; use Synthesis or Verification instead.
- If an earlier intermediate value is recomputed for a new purpose, distinguish the subgoal by purpose, such as "Recompute 130% of 20 for difference check" instead of repeating the exact same subgoal name.
- Avoid generic names such as "Verify result", "Check answer", or "Solve problem" unless that truly is the whole chunk.
- Prefer concrete action + object names such as "Solve for x from the quadratic" or "Substitute radius into the area formula".
- {history_note}
{extra_section}
{last_chunk_note}
- Summary style must be neutral and factual, with no first/second/third-person subject.
- Forbidden starts: "I ...", "We ...", "You ...", "The user ...", "They ...".
- Preferred style: "Computed ...", "Derived ...", "Simplified ...", "Combined ...", "Checked consistency ...".
- cognitive_mode must be one of: {", ".join(COGNITIVE_MODE_CANDIDATES)}
- If none fits, use Other.

Mode selection policy:
- Choose the dominant function of the whole chunk, not the final sentence.
- Use Verification only if the chunk is primarily checking, validating, or rejecting a previously derived candidate, with little or no new derivation.
- If the chunk only rechecks a local arithmetic sub-result or recomputes the same scalar inside the same method, prefer Calculation instead of Verification.
- If substantial new derivation appears before a short final check, label by the derivation, not by the check.
- Do not create a separate Verification identity for a brief confirmation that preserves the same deliverable as the previous chunk.
- If the chunk derives equations, transforms expressions, computes values, solves for unknowns, or simplifies formulas, prefer Calculation or DeepReasoning instead of Verification.
- Use Finalization only when the chunk mainly packages the final answer.
- Use Synthesis only when the chunk mainly combines earlier intermediate results into a conclusion.

Mode definitions:
- ProblemUnderstanding: identify givens, target, notation, or constraints.
- Decomposition: choose a plan, split into subproblems, or introduce cases.
- Exploration: try a possible route without committing to a derivation.
- DeepReasoning: derive a structural relation or logical implication.
- Calculation: carry out algebraic, arithmetic, symbolic, or numeric computation.
- Verification: check a previously obtained result against constraints or the problem statement.
- Reflection: explain why a route fails or needs revision.
- Backtracking: abandon a prior route and restart from an earlier point.
- Synthesis: combine prior intermediate results.
- Finalization: state the final answer in final form.
- Other: use only if none of the above fits.

Problem:
{problem}

Previous steps (cognitive_mode + subgoal + summary):
{history_text}

Merged chunk:
{merged_text}

Output JSON schema:
{{
    "cognitive_mode": "...",
  "subgoal": "...",
    "summary": "2-4 sentence summary preserving all key information in this chunk",
    "outcome": "one-line deliverable from this chunk"
}}
"""


def neutralize_subject_style(text: str) -> str:
    if not text:
        return text
    text = re.sub(
        r"(^|[.!?]\s+)(?:the\s+user|user|they|he|she|we|i|you)\s+",
        r"\1",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_cognitive_mode(mode: str) -> str:
    if not mode:
        return "Other"
    raw = re.sub(r"[^A-Za-z]", "", mode)
    if not raw:
        return "Other"

    lower_map = {m.lower(): m for m in COGNITIVE_MODE_CANDIDATES}
    direct = lower_map.get(raw.lower())
    if direct:
        return direct

    synonyms = {
        "understanding": "ProblemUnderstanding",
        "problemunderstand": "ProblemUnderstanding",
        "decompose": "Decomposition",
        "plan": "Decomposition",
        "explore": "Exploration",
        "reason": "DeepReasoning",
        "compute": "Calculation",
        "calc": "Calculation",
        "verify": "Verification",
        "check": "Verification",
        "reflect": "Reflection",
        "backtrack": "Backtracking",
        "synth": "Synthesis",
        "final": "Finalization",
    }
    raw_lower = raw.lower()
    for key, val in synonyms.items():
        if key in raw_lower:
            return val
    return "Other"


def normalize_subgoal(text: str) -> str:
    text = strip_refine_prefix(text.lower())
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def find_duplicate_step_index(
    steps: List[MergeStep],
    new_subgoal: str,
    profile: SegmentationProfile,
) -> Optional[int]:
    new_norm = normalize_subgoal(new_subgoal)
    if not new_norm:
        return None

    for i in range(len(steps) - 1, -1, -1):
        old_norm = normalize_subgoal(steps[i].subgoal)
        if not old_norm:
            continue
        if new_norm == old_norm:
            return i
        if similarity(new_norm, old_norm) >= profile.duplicate_subgoal_similarity:
            return i
    return None


def is_genuine_refinement(step: MergeStep) -> bool:
    text = " ".join([step.subgoal, step.summary, step.outcome]).lower()
    return any(cue in text for cue in REFINEMENT_CUES)


def has_distinct_method_signal(step: MergeStep) -> bool:
    text = " ".join([step.subgoal, step.summary, step.text]).lower()
    return any(cue in text for cue in METHOD_DISTINCTION_CUES)


def has_packaging_signal(step: MergeStep) -> bool:
    text = " ".join([step.subgoal, step.summary, step.outcome, step.text]).lower()
    return any(cue in text for cue in PACKAGING_CUES)


def has_final_answer_signal(step: MergeStep) -> bool:
    text = " ".join([step.subgoal, step.summary, step.outcome, step.text]).lower()
    return has_packaging_signal(step) or any(cue in text for cue in FINAL_ANSWER_CUES)


def infer_method_suffix(step: MergeStep) -> Optional[str]:
    text = " ".join([step.subgoal, step.summary, step.text]).lower()
    method_map = [
        ("substitution", "via substitution"),
        ("inverse", "via inverse evaluation"),
        ("direct", "via direct evaluation"),
        ("formula", "via formula"),
        ("vector", "via vector method"),
        ("slope", "via slope check"),
        ("shoelace", "via shoelace formula"),
        ("graph", "via graph check"),
        ("table", "via lookup table"),
        ("test", "via test cases"),
        ("example", "via examples"),
    ]
    for key, suffix in method_map:
        if key in text:
            return suffix
    return None


def has_calculation_revisit_signal(step: MergeStep) -> bool:
    text = " ".join([step.subgoal, step.summary, step.text]).lower()
    return any(cue in text for cue in CALCULATION_REVISIT_CUES)


def has_local_check_signal(step: MergeStep) -> bool:
    text = " ".join([step.subgoal, step.summary, step.text]).lower()
    return any(cue in text for cue in LOCAL_CHECK_CUES)


def has_continuation_signal(step: MergeStep) -> bool:
    text = " ".join([step.subgoal, step.summary, step.outcome]).lower()
    return any(cue in text for cue in CONTINUATION_CUES)


def has_initiation_signal(step: MergeStep) -> bool:
    text = " ".join([step.subgoal, step.summary, step.outcome]).lower()
    return any(cue in text for cue in INITIATION_CUES)


def has_micro_calculation_signal(step: MergeStep) -> bool:
    text = " ".join([step.subgoal, step.summary, step.outcome]).lower()
    return any(cue in text for cue in MICRO_CALCULATION_CUES)


def extract_number_tokens(text: str) -> Set[str]:
    return set(re.findall(r"-?\d+(?:/\d+|\.\d+)?", text))


def extract_focus_tokens(text: str) -> Set[str]:
    tokens = re.findall(r"[A-Za-z]+|\d+(?:/\d+|\.\d+)?", text.lower())
    return {
        tok
        for tok in tokens
        if tok not in FOCUS_TOKEN_STOPWORDS and (len(tok) > 2 or bool(re.search(r"\d", tok)))
    }


def shares_anchor_numbers(prev: MergeStep, curr: MergeStep) -> bool:
    prev_tokens = extract_number_tokens(" ".join([prev.subgoal, prev.summary, prev.outcome]))
    curr_tokens = extract_number_tokens(" ".join([curr.subgoal, curr.summary, curr.outcome]))
    return bool(prev_tokens & curr_tokens)


def shares_focus(prev: MergeStep, curr: MergeStep) -> bool:
    prev_tokens = extract_focus_tokens(" ".join([prev.subgoal, prev.summary, prev.outcome]))
    curr_tokens = extract_focus_tokens(" ".join([curr.subgoal, curr.summary, curr.outcome]))
    overlap = prev_tokens & curr_tokens
    if len(overlap) >= 2:
        return True
    if len(overlap) >= 1 and shares_anchor_numbers(prev, curr):
        return True
    return False


def shares_numeric_signature(prev: MergeStep, curr: MergeStep) -> bool:
    prev_tokens = extract_number_tokens(" ".join([prev.summary, prev.outcome, prev.text]))
    curr_tokens = extract_number_tokens(" ".join([curr.summary, curr.outcome, curr.text]))
    if not prev_tokens or not curr_tokens:
        return False
    return prev_tokens == curr_tokens or len(prev_tokens & curr_tokens) >= min(len(prev_tokens), len(curr_tokens))


def get_recent_duplicate_step(
    steps: List[MergeStep],
    new_step: MergeStep,
    profile: SegmentationProfile,
) -> Optional[MergeStep]:
    current = normalize_subgoal(new_step.subgoal)
    if not current:
        return None
    for step in reversed(steps):
        if similarity(normalize_subgoal(step.subgoal), current) >= profile.duplicate_subgoal_similarity:
            return step
    return None


def starts_with_casework_verb(text: str) -> bool:
    lowered = strip_refine_prefix(text.lower())
    return any(lowered.startswith(f"{verb} ") for verb in CASEWORK_VERBS)


def has_casework_signal(step: MergeStep) -> bool:
    text = " ".join([step.subgoal, step.summary, step.outcome]).lower()
    return any(cue in text for cue in CASEWORK_CUES)


def starts_with_exploration_verb(text: str) -> bool:
    lowered = strip_refine_prefix(text.lower())
    return any(lowered.startswith(f"{verb} ") for verb in EXPLORATION_VERBS)


def normalize_casework_subgoal(text: str) -> str:
    text = strip_refine_prefix(text.lower())
    text = re.sub(r"-?\d+(?:/\d+|\.\d+)?", " ", text)
    text = re.sub(
        r"\b(?:candidate|candidates|case|cases|value|values|number|numbers|score|scores|ambiguity|unique|uniqueness|representability|possibility|possibilities)\b",
        " ",
        text,
    )
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_exploration_subgoal(text: str) -> str:
    text = strip_refine_prefix(text.lower())
    text = re.sub(
        r"\b(?:candidate|candidates|hypothesis|hypotheses|approach|approaches|method|methods|route|routes|possibility|possibilities|strategy|strategies)\b",
        " ",
        text,
    )
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def has_model_search_signal(step: MergeStep) -> bool:
    text = " ".join([step.subgoal, step.summary, step.outcome]).lower()
    return any(cue in text for cue in MODEL_SEARCH_CUES)


def normalize_model_search_subgoal(text: str) -> str:
    text = strip_refine_prefix(text.lower())
    text = re.sub(
        r"\b(?:candidate|candidates|model|models|geometry|geometric|folding|folded|net|solid|polyhedron|polyhedral|volume|decomposition|decompose|tetrahedron|tetrahedral|pyramid|pyramidal|prism|prismatic|prismatoid|octahedron|coordinate|coordinates|cross|section|cross sectional|cross sectional|face|faces|vertex|vertices|edge|edges|apex|assembly|structure|configuration)\b",
        " ",
        text,
    )
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_refine_step(step: MergeStep) -> bool:
    return step.subgoal.lower().startswith("refine:")


def should_merge_casework_steps(
    prev: MergeStep,
    curr: MergeStep,
    profile: SegmentationProfile,
) -> bool:
    if not profile.casework_merge:
        return False

    allowed_modes = {"Calculation", "Verification", "Exploration", "Reflection", "Synthesis"}
    if prev.cognitive_mode not in allowed_modes or curr.cognitive_mode not in allowed_modes:
        return False
    if has_packaging_signal(prev) or has_packaging_signal(curr):
        return False
    if has_distinct_method_signal(prev) or has_distinct_method_signal(curr):
        return False

    prev_casework = starts_with_casework_verb(prev.subgoal) or has_casework_signal(prev)
    curr_casework = starts_with_casework_verb(curr.subgoal) or has_casework_signal(curr)
    if not prev_casework or not curr_casework:
        return False

    prev_norm = normalize_casework_subgoal(prev.subgoal)
    curr_norm = normalize_casework_subgoal(curr.subgoal)
    if not prev_norm or not curr_norm:
        return False
    subgoal_sim = similarity(prev_norm, curr_norm)
    if subgoal_sim < profile.casework_subgoal_similarity:
        if not shares_focus(prev, curr):
            return False

    if subgoal_sim >= profile.casework_subgoal_similarity:
        if prev.cognitive_mode == curr.cognitive_mode:
            return True
        if {prev.cognitive_mode, curr.cognitive_mode} <= {"Verification", "Calculation", "Synthesis"}:
            return True

    if shares_focus(prev, curr):
        return True
    if similarity(prev.summary, curr.summary) >= 0.6:
        return True
    if similarity(prev.outcome, curr.outcome) >= 0.6:
        return True
    if prev.cognitive_mode in {"Verification", "Synthesis", "Calculation"} and curr.cognitive_mode in {"Verification", "Synthesis", "Calculation"}:
        if similarity(prev.summary, curr.summary) >= 0.45 or similarity(prev.outcome, curr.outcome) >= 0.45:
            return True
    return False


def should_merge_exploration_steps(
    prev: MergeStep,
    curr: MergeStep,
    profile: SegmentationProfile,
) -> bool:
    if not profile.exploration_merge:
        return False

    allowed_modes = {"Exploration", "Reflection", "Decomposition", "ProblemUnderstanding"}
    if prev.cognitive_mode not in allowed_modes or curr.cognitive_mode not in allowed_modes:
        return False
    if has_packaging_signal(prev) or has_packaging_signal(curr):
        return False
    if has_distinct_method_signal(prev) or has_distinct_method_signal(curr):
        return False

    prev_norm = normalize_exploration_subgoal(prev.subgoal)
    curr_norm = normalize_exploration_subgoal(curr.subgoal)
    if not prev_norm or not curr_norm:
        return False

    if similarity(prev_norm, curr_norm) >= profile.exploration_subgoal_similarity:
        return True

    if starts_with_exploration_verb(prev.subgoal) and starts_with_exploration_verb(curr.subgoal):
        if shares_focus(prev, curr):
            return True
        if similarity(prev.summary, curr.summary) >= 0.48 or similarity(prev.outcome, curr.outcome) >= 0.48:
            return True
        if not extract_number_tokens(prev.text) and not extract_number_tokens(curr.text):
            return True

    return False


def should_merge_model_search_steps(
    prev: MergeStep,
    curr: MergeStep,
    profile: SegmentationProfile,
) -> bool:
    if not profile.model_search_merge:
        return False

    allowed_modes = {"Exploration", "Reflection", "Decomposition", "ProblemUnderstanding", "DeepReasoning"}
    if prev.cognitive_mode not in allowed_modes or curr.cognitive_mode not in allowed_modes:
        return False
    if has_packaging_signal(prev) or has_packaging_signal(curr):
        return False

    if not has_model_search_signal(prev) or not has_model_search_signal(curr):
        return False

    prev_norm = normalize_model_search_subgoal(prev.subgoal)
    curr_norm = normalize_model_search_subgoal(curr.subgoal)
    if prev_norm and curr_norm and similarity(prev_norm, curr_norm) >= profile.model_search_subgoal_similarity:
        return True

    if shares_focus(prev, curr):
        return True
    if similarity(prev.summary, curr.summary) >= 0.42:
        return True
    if similarity(prev.outcome, curr.outcome) >= 0.42:
        return True

    return False


def should_merge_refine_chain(
    prev: MergeStep,
    curr: MergeStep,
    profile: SegmentationProfile,
) -> bool:
    if not profile.aggressive_refine_merge:
        return False
    if not (is_refine_step(prev) or is_refine_step(curr)):
        return False
    if has_packaging_signal(prev) or has_packaging_signal(curr):
        return False

    prev_norm = normalize_subgoal(prev.subgoal)
    curr_norm = normalize_subgoal(curr.subgoal)
    if prev_norm and curr_norm and similarity(prev_norm, curr_norm) >= profile.refine_subgoal_similarity:
        return True

    if shares_focus(prev, curr):
        if shares_numeric_signature(prev, curr):
            return True
        if similarity(prev.summary, curr.summary) >= 0.45 or similarity(prev.outcome, curr.outcome) >= 0.45:
            return True

    if is_refine_step(curr) and prev.cognitive_mode in {"Verification", "Synthesis", "Calculation", "Reflection"}:
        if shares_focus(prev, curr) or similarity(prev.outcome, curr.outcome) >= 0.5:
            return True

    return False


def is_finalization_followup(final_step: MergeStep, other_step: MergeStep) -> bool:
    if final_step.cognitive_mode != "Finalization":
        return False
    if other_step.cognitive_mode not in {"Verification", "Reflection", "Synthesis", "Calculation"}:
        return False
    if has_distinct_method_signal(other_step):
        return False
    if has_final_answer_signal(other_step):
        return True
    if other_step.cognitive_mode in {"Verification", "Reflection"}:
        if shares_focus(final_step, other_step) or shares_numeric_signature(final_step, other_step):
            return True
        return similarity(final_step.outcome, other_step.outcome) >= 0.5
    if other_step.cognitive_mode == "Synthesis":
        if shares_focus(final_step, other_step) or shares_numeric_signature(final_step, other_step):
            return True
        return similarity(final_step.outcome, other_step.outcome) >= 0.58
    if other_step.cognitive_mode == "Calculation":
        if has_calculation_revisit_signal(other_step):
            return True
        if shares_numeric_signature(final_step, other_step):
            return True
        return similarity(final_step.outcome, other_step.outcome) >= 0.72
    return False


def is_lightweight_local_verification(prev: MergeStep, curr: MergeStep) -> bool:
    if prev.cognitive_mode not in {"Calculation", "DeepReasoning"}:
        return False
    if curr.cognitive_mode != "Verification":
        return False
    if has_distinct_method_signal(curr):
        return False
    if not has_local_check_signal(curr):
        return False
    if shares_numeric_signature(prev, curr):
        return True
    return similarity(prev.outcome, curr.outcome) >= 0.85


def is_incremental_same_calculation(prev: MergeStep, curr: MergeStep) -> bool:
    allowed_modes = {"Calculation", "DeepReasoning", "Synthesis", "Verification"}
    if prev.cognitive_mode not in allowed_modes or curr.cognitive_mode not in allowed_modes:
        return False
    if has_packaging_signal(prev) or has_packaging_signal(curr):
        return False
    if has_distinct_method_signal(prev) or has_distinct_method_signal(curr):
        return False

    shared_focus = shares_focus(prev, curr)
    shared_anchor = shares_anchor_numbers(prev, curr)
    prev_norm = normalize_subgoal(prev.subgoal)
    curr_norm = normalize_subgoal(curr.subgoal)
    subgoal_sim = similarity(normalize_subgoal(prev.subgoal), normalize_subgoal(curr.subgoal))
    outcome_sim = similarity(prev.outcome, curr.outcome)

    if prev_norm and prev_norm == curr_norm and has_micro_calculation_signal(prev) and has_micro_calculation_signal(curr):
        if shared_focus or shared_anchor:
            return True

    if prev.cognitive_mode == "Verification" and curr.cognitive_mode == "Verification":
        if has_continuation_signal(curr) and (shared_focus or shared_anchor or subgoal_sim >= 0.6):
            return True

    if prev.cognitive_mode in {"Calculation", "DeepReasoning", "Synthesis"} and curr.cognitive_mode == "Verification":
        if has_local_check_signal(curr):
            return True
        if has_continuation_signal(curr) and (shared_focus or shared_anchor or outcome_sim >= 0.72):
            return True

    if prev.cognitive_mode in {"Calculation", "DeepReasoning", "Synthesis"} and curr.cognitive_mode in {
        "Calculation",
        "DeepReasoning",
        "Synthesis",
    }:
        if has_continuation_signal(curr) and (shared_focus or shared_anchor or subgoal_sim >= 0.6):
            return True
        if has_initiation_signal(prev) and has_micro_calculation_signal(curr) and (shared_focus or shared_anchor):
            return True
        if has_micro_calculation_signal(prev) and has_micro_calculation_signal(curr):
            if shared_focus and (shared_anchor or subgoal_sim >= 0.55 or outcome_sim >= 0.72):
                return True

    return False


def merge_step_content(target: MergeStep, incoming: MergeStep) -> None:
    target.paragraph_indices.extend(incoming.paragraph_indices)
    target.paragraph_indices = sorted(set(target.paragraph_indices))
    target.text = (target.text + "\n\n" + incoming.text).strip()

    if len(incoming.summary) > len(target.summary):
        target.summary = incoming.summary
    if len(incoming.outcome) > len(target.outcome):
        target.outcome = incoming.outcome

    if (
        {target.cognitive_mode, incoming.cognitive_mode} <= {"Finalization", "Reflection", "Verification"}
        and (has_final_answer_signal(target) or has_final_answer_signal(incoming))
    ):
        target.cognitive_mode = "Finalization"
        incoming_text = " ".join([incoming.subgoal, incoming.summary, incoming.outcome, incoming.text]).lower()
        if incoming.cognitive_mode == "Finalization" or "boxed" in incoming_text or "final answer" in incoming_text:
            target.subgoal = incoming.subgoal
        return

    if incoming.cognitive_mode == "Finalization" and target.cognitive_mode != "Finalization":
        if has_final_answer_signal(incoming) or has_final_answer_signal(target):
            target.cognitive_mode = "Finalization"
            target.subgoal = incoming.subgoal
            return

    if target.cognitive_mode in MERGEABLE_FOLLOWUP_MODES and incoming.cognitive_mode not in MERGEABLE_FOLLOWUP_MODES:
        if is_finalization_followup(target, incoming):
            return
        target.cognitive_mode = incoming.cognitive_mode
        if not incoming.subgoal.lower().startswith("refine:"):
            target.subgoal = incoming.subgoal


def make_subgoal_distinct(
    steps: List[MergeStep],
    new_step: MergeStep,
    profile: SegmentationProfile,
) -> None:
    existing = {normalize_subgoal(step.subgoal) for step in steps if step.subgoal}
    current = normalize_subgoal(new_step.subgoal)
    if not current or current not in existing:
        return

    previous = get_recent_duplicate_step(steps, new_step, profile)
    suffix = None
    if new_step.cognitive_mode == "Verification":
        suffix = infer_method_suffix(new_step)
    elif new_step.cognitive_mode in {"Calculation", "DeepReasoning"} and has_calculation_revisit_signal(new_step):
        suffix = "for consistency check"
    elif new_step.cognitive_mode == "Finalization" and has_packaging_signal(new_step):
        suffix = "in boxed format"

    if suffix is None and previous is not None and new_step.cognitive_mode in {"Calculation", "DeepReasoning"}:
        if (
            shares_numeric_signature(previous, new_step)
            or similarity(previous.outcome, new_step.outcome) >= profile.duplicate_outcome_similarity
        ):
            suffix = "for consistency check"

    if suffix and suffix not in new_step.subgoal.lower():
        new_step.subgoal = f"{new_step.subgoal} {suffix}"


def normalize_step_sequence(steps: List[MergeStep]) -> List[MergeStep]:
    for idx, step in enumerate(steps[:-1]):
        if step.cognitive_mode != "Finalization":
            continue

        later_steps = steps[idx + 1 :]
        later_substantive = any(
            later.cognitive_mode != "Finalization" and not is_finalization_followup(step, later)
            for later in later_steps
        )
        if later_substantive:
            step.cognitive_mode = "Synthesis"
            if any(token in step.subgoal.lower() for token in ["final answer", "boxed", "state final", "present final"]):
                step.subgoal = "Summarize established result"

    if steps and steps[-1].cognitive_mode != "Finalization" and has_final_answer_signal(steps[-1]):
        steps[-1].cognitive_mode = "Finalization"
        if steps[-1].subgoal.lower().startswith(("compute ", "derive ", "combine ", "solve ", "summarize ")):
            steps[-1].subgoal = "State final answer with brief verification summary"

    for idx, step in enumerate(steps, start=1):
        step.step_id = idx

    return steps


def should_merge_duplicate_neighbors(
    prev: MergeStep,
    curr: MergeStep,
    profile: SegmentationProfile,
) -> bool:
    prev_norm = normalize_subgoal(prev.subgoal)
    curr_norm = normalize_subgoal(curr.subgoal)
    if not prev_norm or not curr_norm:
        return False

    subgoal_sim = similarity(prev_norm, curr_norm)
    summary_sim = similarity(prev.summary, curr.summary)
    text_sim = similarity(prev.text, curr.text)
    outcome_sim = similarity(prev.outcome, curr.outcome)

    if prev.cognitive_mode == "Finalization" and curr.cognitive_mode != "Finalization":
        return is_finalization_followup(prev, curr)

    if curr.cognitive_mode == "Finalization" and prev.cognitive_mode != "Finalization":
        return is_finalization_followup(curr, prev)

    if prev.cognitive_mode == "Finalization" and curr.cognitive_mode == "Finalization":
        if has_final_answer_signal(prev) or has_final_answer_signal(curr):
            return outcome_sim >= 0.65 or summary_sim >= 0.62 or shares_numeric_signature(prev, curr)
        return outcome_sim >= 0.75 or summary_sim >= 0.7 or subgoal_sim >= 0.92

    if (
        {prev.cognitive_mode, curr.cognitive_mode} <= {"Finalization", "Reflection", "Verification"}
        and has_final_answer_signal(prev)
        and has_final_answer_signal(curr)
    ):
        return outcome_sim >= 0.7 or summary_sim >= 0.7 or shares_numeric_signature(prev, curr)

    if is_lightweight_local_verification(prev, curr):
        return True

    if has_distinct_method_signal(prev) or has_distinct_method_signal(curr):
        return False

    if should_merge_refine_chain(prev, curr, profile):
        return True

    if should_merge_casework_steps(prev, curr, profile):
        return True

    if should_merge_model_search_steps(prev, curr, profile):
        return True

    if should_merge_exploration_steps(prev, curr, profile):
        return True

    if is_incremental_same_calculation(prev, curr):
        return True

    if prev.cognitive_mode != curr.cognitive_mode:
        return False

    if prev.cognitive_mode in {"Calculation", "DeepReasoning"}:
        if subgoal_sim >= 0.95 and (outcome_sim >= 0.85 or shares_numeric_signature(prev, curr)):
            return True

    if subgoal_sim >= 0.97:
        return summary_sim >= 0.72 or text_sim >= 0.72

    if subgoal_sim >= 0.9 and curr.subgoal.lower().startswith("refine:"):
        return summary_sim >= 0.8 or text_sim >= 0.82

    if outcome_sim >= 0.95 and summary_sim >= 0.88:
        return True

    return False


def should_merge_adjacent_steps(
    prev: MergeStep,
    curr: MergeStep,
    profile: SegmentationProfile,
) -> bool:
    return should_merge_duplicate_neighbors(prev, curr, profile)


def collapse_redundant_steps(
    steps: List[MergeStep],
    profile: SegmentationProfile,
) -> List[MergeStep]:
    if not steps:
        return steps

    collapsed = [steps[0]]
    for step in steps[1:]:
        prev = collapsed[-1]
        if should_merge_adjacent_steps(prev, step, profile):
            merge_step_content(prev, step)
            continue
        collapsed.append(step)

    for i, step in enumerate(collapsed, start=1):
        step.step_id = i

    return normalize_step_sequence(collapsed)


def judge_pair(
    profile: SegmentationProfile,
    api_key: str,
    model: str,
    problem: str,
    paragraphs: List[str],
    idx: int,
    temperature: float,
    max_tokens: Optional[int],
) -> PairDecision:
    prompt = boundary_prompt(profile, problem, paragraphs[idx], paragraphs[idx + 1])
    raw = call_deepseek(
        api_key,
        model,
        profile.system_boundary,
        prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    try:
        obj = extract_json(raw)
    except Exception as e:
        raise RuntimeError(f"rename_groups extract_json failed: {e}; raw={raw[:1000]}")

    decision = str(obj.get("decision", "NO")).strip().upper()
    if decision not in {"YES", "NO"}:
        decision = "NO"

    try:
        conf = float(obj.get("confidence", 0.5))
    except Exception:
        conf = 0.5

    reason = str(obj.get("reason", ""))
    return PairDecision(idx, idx + 1, decision, conf, reason)


def serial_refine_merge(
    paragraphs: List[str],
    pair_map: Dict[Tuple[int, int], PairDecision],
) -> List[List[int]]:
    groups: List[List[int]] = []
    current = [0]
    for i in range(len(paragraphs) - 1):
        d = pair_map[(i, i + 1)].decision
        if d == "YES":
            current.append(i + 1)
        else:
            groups.append(current)
            current = [i + 1]
    groups.append(current)
    return groups


def serialize_groups_for_output(profile: SegmentationProfile, groups: List[List[int]], steps: List[MergeStep]) -> List[List[int]]:
    if profile.align_groups_to_steps:
        return [list(step.paragraph_indices) for step in steps]
    return groups


def has_successful_segmentation_result(data: Dict) -> bool:
    result = data.get("segmentation_result")
    if not isinstance(result, dict):
        return False
    if result.get("status") == "error":
        return False
    if result.get("error"):
        return False
    required_keys = {"paragraphs", "pair_decisions", "groups", "steps"}
    return required_keys.issubset(result.keys())


def rename_groups(
    profile: SegmentationProfile,
    api_key: str,
    model: str,
    problem: str,
    paragraphs: List[str],
    groups: List[List[int]],
    temperature: float,
    max_tokens: Optional[int],
) -> List[MergeStep]:
    steps: List[MergeStep] = []

    for group_idx, g in enumerate(groups):
        merged_text = "\n\n".join(paragraphs[i] for i in g)
        prompt = rename_prompt(
            profile,
            problem,
            merged_text,
            steps,
            is_last_chunk=group_idx == len(groups) - 1,
        )
        raw = call_deepseek(
            api_key,
            model,
            profile.system_rename,
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        try:
            obj = extract_json(raw)
        except Exception as e:
            raise RuntimeError(f"rename_groups extract_json failed: {e}; raw={raw[:1000]}")

        cognitive_mode = normalize_cognitive_mode(str(obj.get("cognitive_mode", "Other")).strip())
        subgoal = str(obj.get("subgoal", "Unspecified subgoal")).strip()
        summary = str(obj.get("summary", "")).strip()
        outcome = str(obj.get("outcome", "")).strip()

        if not summary:
            summary = merged_text[:900].strip()
        if not outcome:
            outcome = summary.split("\n")[0].strip()

        summary = neutralize_subject_style(summary)
        outcome = neutralize_subject_style(outcome)

        new_step = MergeStep(
            step_id=len(steps) + 1,
            paragraph_indices=list(g),
            cognitive_mode=cognitive_mode,
            subgoal=subgoal,
            summary=summary,
            outcome=outcome,
            text=merged_text,
        )

        dup_idx = find_duplicate_step_index(steps, new_step.subgoal, profile)
        if dup_idx is not None:
            prev = steps[dup_idx]
            if dup_idx == len(steps) - 1 and should_merge_adjacent_steps(prev, new_step, profile):
                merge_step_content(prev, new_step)
                continue

            if is_genuine_refinement(new_step) and not new_step.subgoal.lower().startswith("refine:"):
                new_step.subgoal = make_refine_subgoal(prev.subgoal)

        make_subgoal_distinct(steps, new_step, profile)

        steps.append(new_step)

    return collapse_redundant_steps(steps, profile)


def run_pipeline(
    profile: SegmentationProfile,
    problem: str,
    cot: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: Optional[int],
    auto_merge_threshold: Optional[int],
    auto_merge_group_size: Optional[int],
) -> Dict:
    paragraphs = auto_merge_paragraphs(
        split_paragraphs(cot),
        auto_merge_threshold,
        auto_merge_group_size,
    )
    if len(paragraphs) < 2:
        return {
            "paragraphs": paragraphs,
            "pair_decisions": [],
            "groups": [[0]] if paragraphs else [],
            "steps": [],
        }

    pair_results: List[PairDecision] = []
    for i in range(len(paragraphs) - 1):
        pair_results.append(
            judge_pair(
                profile=profile,
                api_key=api_key,
                model=model,
                problem=problem,
                paragraphs=paragraphs,
                idx=i,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )

    pair_results.sort(key=lambda x: x.left_index)
    pair_map = {(d.left_index, d.right_index): d for d in pair_results}

    groups = serial_refine_merge(paragraphs, pair_map)
    steps = rename_groups(
        profile,
        api_key,
        model,
        problem,
        paragraphs,
        groups,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return {
        "paragraphs": paragraphs,
        "pair_decisions": [asdict(x) for x in pair_results],
        "groups": serialize_groups_for_output(profile, groups, steps),
        "steps": [asdict(x) for x in steps],
    }


def process_sample(
    profile: SegmentationProfile,
    line: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: Optional[int],
    auto_merge_threshold: Optional[int],
    auto_merge_group_size: Optional[int],
) -> Dict:
    data = json.loads(line)
    problem = data["prompt"].split("Please reason step by step")[0].strip()
    cot = data["reasoning_content"].strip()

    result = run_pipeline(
        profile=profile,
        problem=problem,
        cot=cot,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        auto_merge_threshold=auto_merge_threshold,
        auto_merge_group_size=auto_merge_group_size,
    )
    if profile.preserve_error_schema:
        result.setdefault("status", "ok")
        result.setdefault("error", None)

    data["segmentation_result"] = result
    return data


def safe_process_sample(
    profile: SegmentationProfile,
    line: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: Optional[int] = None,
    auto_merge_threshold: Optional[int] = None,
    auto_merge_group_size: Optional[int] = None,
) -> Dict:
    try:
        return process_sample(
            profile=profile,
            line=line,
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            auto_merge_threshold=auto_merge_threshold,
            auto_merge_group_size=auto_merge_group_size,
        )
    except Exception as e:
        raw = None
        try:
            raw = json.loads(line)
            sample_id = raw.get("id")
        except Exception:
            sample_id = None

        if profile.inline_error_result and isinstance(raw, dict):
            error_result = {
                "status": "error",
                "error": str(e),
            }
            if profile.preserve_error_schema:
                paragraphs = auto_merge_paragraphs(
                    split_paragraphs(str(raw.get("reasoning_content", "")).strip()),
                    auto_merge_threshold,
                    auto_merge_group_size,
                )
                error_result.update(
                    {
                        "paragraphs": paragraphs,
                        "pair_decisions": [],
                        "groups": [],
                        "steps": [],
                    }
                )
            raw["segmentation_result"] = {
                **error_result,
            }
            return raw

        return {
            "id": sample_id,
            "error": str(e),
            "raw_line": line.rstrip("\n"),
        }


def load_existing_ids(output_jsonl: str) -> Set[str]:
    existing_ids: Set[str] = set()
    if not os.path.exists(output_jsonl):
        return existing_ids

    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            row = line.strip()
            if not row:
                continue
            try:
                data = json.loads(row)
            except Exception:
                continue
            if "id" in data and data["id"] is not None and has_successful_segmentation_result(data):
                existing_ids.add(str(data["id"]))
    return existing_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
        help="Input JSONL file, one sample per line",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="boundary_result.jsonl",
        help="Output JSONL file",
    )
    parser.add_argument("--model", type=str, default="deepseek-chat")
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of samples processed in parallel",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--api_key_env", type=str, default="DEEPSEEK_API_KEY")
    parser.add_argument("--env_file", type=str, default=".env")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Only process the first N non-empty lines from input_jsonl",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If set, skip samples whose id already exists in output_jsonl",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="max_tokens passed to DeepSeek API",
    )
    parser.add_argument(
        "--auto_merge_threshold",
        type=int,
        default=None,
        help="If set, only trigger paragraph auto-merge when the split paragraph count exceeds this threshold.",
    )
    parser.add_argument(
        "--auto_merge_group_size",
        type=int,
        default=None,
        help="When auto-merge is triggered, merge every N split paragraphs into one larger paragraph.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional dataset name used for auto-selecting a segmentation profile",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="auto",
        choices=["auto", *sorted(PROFILES.keys())],
        help="Segmentation profile; auto infers from --dataset or input path",
    )
    args = parser.parse_args()

    merge_args = (args.auto_merge_threshold, args.auto_merge_group_size)
    if any(value is not None for value in merge_args) and not all(value is not None for value in merge_args):
        raise ValueError("--auto_merge_threshold and --auto_merge_group_size must be provided together")
    if args.auto_merge_threshold is not None and args.auto_merge_threshold <= 0:
        raise ValueError("--auto_merge_threshold must be a positive integer when provided")
    if args.auto_merge_group_size is not None and args.auto_merge_group_size <= 0:
        raise ValueError("--auto_merge_group_size must be a positive integer when provided")

    profile = resolve_profile(args.profile, args.dataset, args.input_jsonl)
    print(f"Using profile: {profile.name}")
    if args.auto_merge_threshold is not None:
        print(
            f"Using auto_merge: threshold>{args.auto_merge_threshold}, group_size={args.auto_merge_group_size}"
        )

    load_env_file(args.env_file)
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing API key in env var: {args.api_key_env} "
            f"(checked env file: {args.env_file})"
        )

    if args.resume:
        existing_ids = load_existing_ids(args.output_jsonl)
        print(f"Loaded existing ids: {len(existing_ids)}")
    else:
        existing_ids = set()
        print("Resume disabled: will process all input samples")

    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        raw_lines = [line for line in f if line.strip()]

    if args.max_samples is not None:
        raw_lines = raw_lines[:args.max_samples]

    lines_to_process = []
    skipped = 0

    for line in raw_lines:
        try:
            data = json.loads(line)
        except Exception:
            lines_to_process.append(line)
            continue

        sample_id = data.get("id")
        if sample_id is not None and str(sample_id) in existing_ids:
            skipped += 1
            continue

        lines_to_process.append(line)

    print(f"Total input samples: {len(raw_lines)}")
    print(f"Skipped existing ids: {skipped}")
    print(f"Need processing: {len(lines_to_process)}")

    output_mode = "a" if args.resume else "w"
    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
    with open(args.output_jsonl, output_mode, encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [
                ex.submit(
                    safe_process_sample,
                    profile,
                    line,
                    api_key,
                    args.model,
                    args.temperature,
                    args.max_tokens,
                    args.auto_merge_threshold,
                    args.auto_merge_group_size,
                )
                for line in lines_to_process
            ]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing samples"):
                item = future.result()
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                fout.flush()

                if "id" in item and item["id"] is not None:
                    existing_ids.add(str(item["id"]))

    print(f"Saved/Appended to: {args.output_jsonl}")
    print(f"Processed this run: {len(lines_to_process)}")
    print(f"Skipped this run: {skipped}")
    print(f"Total known ids now: {len(existing_ids)}")


if __name__ == "__main__":
    main()