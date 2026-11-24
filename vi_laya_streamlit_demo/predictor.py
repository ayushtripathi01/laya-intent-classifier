"""Inference adapter for the locally trained Vi Laya decision model."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
DEFAULT_MODEL_DIR = REPO_ROOT / "vi_laya_checkpoint" / "vi-laya-s0"
DEFAULT_FAMILIES_FILE = APP_DIR / "config" / "families.json"
DEFAULT_INTENTS_FILE = APP_DIR / "config" / "intents.json"


def _json_safe(value: Any) -> Any:
    """Convert common SDK/numpy values into values Streamlit can display."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


class LayaViClassifier:
    """Load the checkpoint and classify with family-specific intent options."""

    def __init__(
        self,
        model_dir: str | Path | None = None,
        questions_file: str | Path | None = None,
        families_file: str | Path | None = None,
        intents_file: str | Path | None = None,
    ) -> None:
        model_dir = Path(
            model_dir or os.getenv("LAYA_MODEL_DIR", str(DEFAULT_MODEL_DIR))
        ).expanduser().resolve()
        questions_file = Path(
            questions_file or os.getenv("LAYA_QUESTIONS_FILE", str(model_dir / "questions.json"))
        ).expanduser().resolve()
        families_file = Path(
            families_file or os.getenv("LAYA_FAMILIES_FILE", str(DEFAULT_FAMILIES_FILE))
        ).expanduser().resolve()
        intents_file = Path(
            intents_file or os.getenv("LAYA_INTENTS_FILE", str(DEFAULT_INTENTS_FILE))
        ).expanduser().resolve()

        required = [
            model_dir / "model.safetensors",
            model_dir / "encoder" / "config.json",
            model_dir / "tokenizer" / "tokenizer.json",
            model_dir / "rl_agent_config.json",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("Checkpoint is incomplete; missing: " + ", ".join(missing))
        if not questions_file.is_file():
            raise FileNotFoundError(f"Questions configuration not found: {questions_file}")
        for catalog in (families_file, intents_file):
            if not catalog.is_file():
                raise FileNotFoundError(f"Intent taxonomy file not found: {catalog}")

        with questions_file.open(encoding="utf-8") as stream:
            self.config = json.load(stream)
        with families_file.open(encoding="utf-8") as stream:
            families = json.load(stream)
        with intents_file.open(encoding="utf-8") as stream:
            intents = json.load(stream)
        self._apply_family_intent_catalog(self.config, families, intents)
        self._validate_questions(self.config)
        self.family_intents = self.config["intents"]
        self.intent_count = sum(len(question["criteria"]) for question in self.family_intents.values())
        self.families_file = families_file
        self.intents_file = intents_file
        with (model_dir / "rl_agent_config.json").open(encoding="utf-8") as stream:
            self.calibration = json.load(stream)
        self.model_dir = model_dir
        self.questions_file = questions_file

        # Laya 0.3.x's documented local-checkpoint API loads the model weights,
        # encoder, tokenizer and temperature calibration from this directory.
        import laya

        loader = getattr(laya, "load", None)
        if not callable(loader):
            raise RuntimeError(
                "The installed Laya package does not expose laya.load(); "
                "install the pinned legacy Laya version in requirements.txt."
            )
        self.agent = loader(str(model_dir))

    @staticmethod
    def _apply_family_intent_catalog(
        config: dict[str, Any], families: Any, intents: Any
    ) -> None:
        """Build the runtime questions from the dataset's canonical catalogs."""
        if not isinstance(families, list) or not families:
            raise ValueError("families.json must be a non-empty list")
        if not isinstance(intents, list) or not intents:
            raise ValueError("intents.json must be a non-empty list")

        family_names: list[str] = []
        intent_to_family: dict[str, str] = {}
        family_criteria: dict[str, str] = {}
        for family in families:
            if not isinstance(family, dict):
                raise ValueError("Each families.json entry must be an object")
            name = family.get("name")
            description = family.get("description")
            member_intents = family.get("intents")
            if not isinstance(name, str) or not name:
                raise ValueError("Each family needs a non-empty name")
            if name in family_criteria:
                raise ValueError(f"Duplicate family in families.json: {name}")
            if not isinstance(description, str) or not description.strip():
                raise ValueError(f"Family {name!r} needs a description")
            if not isinstance(member_intents, list) or not member_intents:
                raise ValueError(f"Family {name!r} needs a non-empty intents list")
            if family.get("n_intents") != len(member_intents):
                raise ValueError(f"Family {name!r} has an incorrect n_intents value")
            family_names.append(name)
            family_criteria[name] = description
            for intent_name in member_intents:
                if not isinstance(intent_name, str) or not intent_name:
                    raise ValueError(f"Family {name!r} contains an invalid intent name")
                if intent_name in intent_to_family:
                    raise ValueError(f"Intent is assigned more than once: {intent_name}")
                intent_to_family[intent_name] = name

        intent_definitions: dict[str, dict[str, Any]] = {}
        for definition in intents:
            if not isinstance(definition, dict):
                raise ValueError("Each intents.json entry must be an object")
            name = definition.get("name")
            option_text = definition.get("option_text")
            if not isinstance(name, str) or not name:
                raise ValueError("Each intent definition needs a non-empty name")
            if name in intent_definitions:
                raise ValueError(f"Duplicate intent definition: {name}")
            if not isinstance(option_text, str) or not option_text.strip():
                raise ValueError(f"Intent {name!r} needs a non-empty option_text")
            expected_family = intent_to_family.get(name)
            if expected_family is None:
                raise ValueError(f"Intent {name!r} is not assigned in families.json")
            if definition.get("family") != expected_family:
                raise ValueError(
                    f"Intent {name!r} is assigned to {definition.get('family')!r} "
                    f"in intents.json but {expected_family!r} in families.json"
                )
            intent_definitions[name] = definition

        missing_definitions = set(intent_to_family) - set(intent_definitions)
        unassigned_definitions = set(intent_definitions) - set(intent_to_family)
        if missing_definitions or unassigned_definitions:
            details = []
            if missing_definitions:
                details.append("missing definitions: " + ", ".join(sorted(missing_definitions)))
            if unassigned_definitions:
                details.append("unassigned definitions: " + ", ".join(sorted(unassigned_definitions)))
            raise ValueError("Family and intent catalogs disagree (" + "; ".join(details) + ")")

        if "family" not in config or not isinstance(config["family"], dict):
            raise ValueError("Checkpoint questions must define the family question")
        saved_family_names = set(config["family"].get("criteria", {}))
        if saved_family_names and saved_family_names != set(family_names):
            raise ValueError("families.json family names do not match the checkpoint question")

        intent_instructions = config.get("intent", {}).get(
            "instructions", "Which specific customer intent does this message express?"
        )
        config["family"]["criteria"] = family_criteria
        config.pop("intent", None)
        config["intents"] = {
            family["name"]: {
                "type": "choice",
                "instructions": intent_instructions,
                "criteria": {
                    intent_name: intent_definitions[intent_name]["option_text"]
                    for intent_name in family["intents"]
                },
            }
            for family in families
        }

    @staticmethod
    def _validate_questions(config: dict[str, Any]) -> None:
        if not isinstance(config, dict) or "family" not in config:
            raise ValueError("questions.json must contain a 'family' choice question")

        def validate_choice(question: Any, question_id: str) -> None:
            if not isinstance(question, dict) or question.get("type") != "choice":
                raise ValueError(f"Question {question_id!r} must be a choice question")
            if not isinstance(question.get("instructions"), str) or not question["instructions"].strip():
                raise ValueError(f"Question {question_id!r} needs non-empty instructions")
            criteria = question.get("criteria")
            if not isinstance(criteria, dict) or not criteria:
                raise ValueError(f"Question {question_id!r} needs a non-empty criteria mapping")

        validate_choice(config["family"], "family")

        # Support the full hierarchical form:
        #   intents: {family_name: {type, instructions, criteria}}
        # A legacy flat `intent` key is accepted for audit compatibility only;
        # prediction deliberately does not use that shared list.
        if "intent" not in config and "intents" not in config:
            raise ValueError("questions.json must contain family-specific 'intents'")
        if "intent" in config:
            validate_choice(config["intent"], "intent")
        if "intents" in config:
            by_family = config["intents"]
            if not isinstance(by_family, dict):
                raise ValueError("'intents' must map family names to choice questions")
            known_families = set(config["family"]["criteria"])
            unknown = set(by_family) - known_families
            if unknown:
                raise ValueError(f"Intent mapping has unknown families: {', '.join(sorted(unknown))}")
            for family_name, question in by_family.items():
                validate_choice(question, f"intents.{family_name}")

    @staticmethod
    def _confidence(answer: dict[str, Any]) -> float:
        # `answer_confidence` is max(p): the probability assigned to the
        # selected choice and the quantity temperature calibration fits.
        # Laya's `confidence` is normalized entropy for choice answers, so it
        # is not comparable across option counts and must not drive this gate.
        confidence = answer.get("answer_confidence", answer.get("confidence"))
        if confidence is None:
            raise ValueError("Laya response did not include a confidence value")
        value = float(confidence)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Laya returned an invalid confidence: {value}")
        return value

    def _predict_one(
        self, utterance: str, question_id: str, question_config: dict[str, Any]
    ) -> tuple[str, float, Any]:
        raw = self.agent.predict(utterance, {question_id: question_config})
        answer = raw["answers"][question_id]
        choice = answer.get("choice")
        if choice not in question_config["criteria"]:
            raise ValueError(f"Laya returned unknown {question_id} choice: {choice!r}")
        return str(choice), self._confidence(answer), raw

    def predict(self, utterance: str, threshold: float = 0.80) -> dict[str, Any]:
        if not isinstance(utterance, str) or not utterance.strip():
            raise ValueError("Enter a non-empty customer message.")
        threshold = float(threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0 and 1.")

        family, family_confidence, family_raw = self._predict_one(
            utterance.strip(), "family", self.config["family"]
        )
        if family == "out_of_scope":
            intent = "out_of_scope"
            intent_confidence = 1.0
            intent_raw = None
            combined_confidence = family_confidence
            intent_mapping_available = True
            decision = "AUTO_HANDLE" if combined_confidence >= threshold else "HUMAN_REVIEW"
            mapping_message = None
        else:
            intent_question = self.family_intents.get(family)
            if intent_question is None:
                # Fail closed: never compare against a global catalog when the
                # predicted family has no mapped intent list.
                intent = None
                intent_confidence = None
                intent_raw = None
                combined_confidence = None
                intent_mapping_available = False
                decision = "HUMAN_REVIEW"
                mapping_message = (
                    f"No family-specific intent options are configured for {family!r}. "
                    "Stage 2 was skipped and this message needs human review."
                )
            else:
                intent, intent_confidence, intent_raw = self._predict_one(
                    utterance.strip(), "intent", intent_question
                )
                combined_confidence = family_confidence * intent_confidence
                intent_mapping_available = True
                decision = "AUTO_HANDLE" if combined_confidence >= threshold else "HUMAN_REVIEW"
                mapping_message = None

        return {
            "family": family,
            "intent": intent,
            "family_confidence": family_confidence,
            "intent_confidence": intent_confidence,
            "confidence": combined_confidence,
            "decision": decision,
            "threshold": threshold,
            "intent_mapping_available": intent_mapping_available,
            "mapping_message": mapping_message,
            "raw": _json_safe({"family": family_raw, "intent": intent_raw}),
        }
