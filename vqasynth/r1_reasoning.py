import time
import base64
import io
import json
import random
from PIL import Image
from openai import OpenAI

class R1Reasoner:
    def __init__(
        self,
        api_key: str,
        model: str,
        image_column: str,
        text_column: str,
        delay: int = 1
    ):
        """
        Args:
            api_key: API key for OpenAI (used by your custom OpenAI client or official library).
            model: Model name (e.g., 'gpt-4').
            image_column: The name of the column where images are stored.
            text_column: The name of the column containing conversation data (list of dicts).
            delay: Minimum time (seconds) to wait between calls.
        """
        self.model = model
        self.image_column = image_column
        self.text_column = text_column
        self.delay = delay

        self.client = OpenAI(api_key=api_key)

    def encode_image(self, image):
        """
        Encodes an image as base64. Handles either a file path (str) or a PIL.Image.
        Returns a base64-encoded string.
        """
        if isinstance(image, str):
            with open(image, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode("utf-8")
            return encoded
        elif isinstance(image, Image.Image):
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return encoded
        return ""

    def _extract_qa_pairs(self, conversation):
        """
        Extract all user->assistant Q/A pairs from the conversation.
        Returns a list of (question_str, answer_str) tuples.
        """
        pairs = []
        for i in range(len(conversation) - 1):
            current_msg = conversation[i]
            next_msg = conversation[i + 1]
            # Check if current is user, next is assistant
            if current_msg.get("role") == "user" and next_msg.get("role") == "assistant":
                question_texts = [
                    c.get("text", "") for c in current_msg.get("content", [])
                    if c.get("type") == "text" and c.get("text") is not None
                ]
                answer_texts = [
                    c.get("text", "") for c in next_msg.get("content", [])
                    if c.get("type") == "text" and c.get("text") is not None
                ]
                if question_texts and answer_texts:
                    # Take the first text if multiple appear
                    pairs.append((question_texts[0], answer_texts[0]))
        return pairs

    def _find_random_qa_pair(self, conversation):
        """
        Finds a random user->assistant Q/A pair from the conversation.
        Returns (question_str, answer_str) or (None, None) if none found.
        """
        pairs = self._extract_qa_pairs(conversation)
        if not pairs:
            return None, None
        qa = random.choice(pairs)
        return qa

    def _format_input_string(self, question):
        """
        Format the question with special tokens into your desired string for 'input'.
        Example format:
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>  detailed thinking on<|eot_id|><|start_header_id|>user<|end_header_id|>  {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
        input_str = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>  detailed thinking on"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>  "
            f"{question}"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        )
        return input_str

    def run(self, question: str, answer: str, image):
        """
        Build a prompt that includes the question/answer pair,
        call the model, and return the chain-of-thought or reasoning.

        Args:
            question: The question from user
            answer: The assistant's answer
            image: The image data (path or PIL image)
        Returns:
            reasoning (str)
        """
        # Encode image
        base64_image = self.encode_image(image)

        # Example system prompt
        system_prompt = (
            "Please see the quantitative distance question-answer pair given and use the remaining information "
            "to formulate a reasoning trace to support the answer. Some information may be partially incorrect, "
            "so you will need to reason about the scene and its scale to ensure high-quality, consistent, and robust responses. "
            "Make sure to generate a CoT trace in the first-person voice, as though revealing an internal monologue, "
            "inside of <think> tokens before providing the final answer inside <answer> tokens. Don't refer to the information "
            "as if it were stated, mentioned, previously referenced, or provided, but rather inferred or directly observed as if "
            "you were an embodied AI tasked with making these judgements after some thought about what you see in the scene. "
            "Feel free to simplify the way objects in the scene are referenced for a more colloquial description.\n"
        )

        qa_pair = f"Question: {question} Answer: {answer}"
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": system_prompt + qa_pair},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]}
            ]
        )

        reasoning = response.choices[0].message.content
        return reasoning

    def apply_transform(self, examples, images=None, text=None):
        """
        Called by dataset.map(..., batched=True).
        """
        images = images or self.image_column
        text = text or self.text_column

        is_batched = isinstance(examples[images], list)

        if not is_batched:
            # Single example scenario
            conversation = examples[text]
            image_data = examples[images]
            if isinstance(image_data, list) and len(image_data) > 0:
                image_data = image_data[0]

            question, ans = self._find_random_qa_pair(conversation)

            if question is None or ans is None:
                examples["input"] = None
                examples["output"] = None
                examples["reasoning"] = None
            else:
                input_string = self._format_input_string(question)
                examples["input"] = input_string
                reasoning_str = self.run(question, ans, image_data)
                examples["output"] = reasoning_str
                examples["reasoning"] = "on"

            return examples

        else:
            conversation_batch = examples[text]
            image_batch = examples[images]

            inputs = []
            outputs = []
            reasonings = []

            for idx, (conv, img) in enumerate(zip(conversation_batch, image_batch)):
                if isinstance(img, list) and len(img) > 0:
                    img = img[0]

                question, ans = self._find_random_qa_pair(conv)

                if question is None or ans is None:
                    inputs.append(None)
                    outputs.append(None)
                    reasonings.append(None)
                    continue

                # Create the formatted input
                input_string = self._format_input_string(question)
                inputs.append(input_string)

                # Get reasoning from the model
                reasoning_str = self.run(question, ans, img)
                outputs.append(reasoning_str)
                reasonings.append("on")

            examples["input"] = inputs
            examples["output"] = outputs
            examples["reasoning"] = reasonings

            return examples


class SoftTraceReasoner(R1Reasoner):
    """Phase-3 teacher: grounded + uncertainty-preserving reasoning traces.

    Extends :class:`R1Reasoner` by injecting the per-scene *grounded state* the
    pipeline now surfaces — object labels with intrinsics-calibrated metric
    positions (metric calibration), per-object depth confidence (surfaced
    ``depth_conf``), and pairwise cross-view match reliability (surfaced
    ``inlier_ratio``) — into the teacher prompt, and steering the teacher to
    weave that state into a first-person trace that *carries calibrated
    confidence* rather than hard commitments. Low-confidence measurements are
    rendered as rough/hedged so the student learns soft composition instead of
    over-committing on ungrounded geometry.

    ``grounded_state`` schema (JSON-serializable dict or its JSON string; every
    field optional so partial records degrade gracefully)::

        {
          "frames": int,
          "objects": [{"label": str, "center_m": [x, y, z],
                       "depth_conf": float | None, "frame_ids": [int]}],
          "pairs":   [{"a": str, "b": str, "distance_m": float,
                       "inlier_ratio": float | None}],
        }

    Back-compat: when no grounded state is available for a row, this behaves
    exactly like :class:`R1Reasoner` (same system prompt, same output), so it is
    a drop-in replacement in the ``r1_reasoning`` stage.
    """

    SOFT_SYSTEM_PROMPT = (
        "You are an embodied AI reasoning about a scene you are directly observing. You are given a "
        "quantitative spatial question and its answer, together with your own grounded observations of the "
        "scene: object positions in real-world meters and how sure you are of each (some measurements are "
        "reliable, others are rough). Formulate a chain-of-thought in the first-person voice, as an internal "
        "monologue, inside <think> tokens, then give the final answer inside <answer> tokens. Weave your "
        "grounded observations into the reasoning, but PROPAGATE YOUR CONFIDENCE rather than flattening it: "
        "lean on the measurements you are confident about, and where a measurement is low-confidence, hedge it, "
        "cross-check it against other cues, or treat it as approximate instead of committing to it as if certain. "
        "Do not refer to the information as if it were stated, mentioned, or provided — reason as if inferred or "
        "directly observed. Feel free to reference objects colloquially.\n"
    )

    # SenseNova-style perspective-taking: questions posed from a viewpoint other
    # than the camera's own need the frame of reference re-anchored before
    # left/right/front/behind are judged (relations flip under rotation).
    _PERSPECTIVE_MARKERS = (
        "from where", "from the perspective", "from the viewpoint", "if you were standing",
        "if you were at", "if you stood", "from camera", "from the camera", "facing",
        "egocentric", "allocentric", "from behind", "standing at", "looking from",
    )
    PERSPECTIVE_ADDENDUM = (
        "This question asks you to reason from a specific viewpoint or frame of reference, not necessarily your "
        "own. Before judging left/right/front/behind, explicitly re-anchor to that viewpoint and transform the "
        "scene into it — spatial relations can flip when the observer's orientation changes.\n"
    )

    @classmethod
    def _is_perspective_question(cls, question):
        q = (question or "").lower()
        return any(marker in q for marker in cls._PERSPECTIVE_MARKERS)

    def __init__(
        self,
        api_key: str,
        model: str,
        image_column: str,
        text_column: str,
        delay: int = 1,
        grounded_state_column: str = "grounded_state",
    ):
        super().__init__(api_key, model, image_column, text_column, delay=delay)
        self.grounded_state_column = grounded_state_column

    @staticmethod
    def _confidence_phrase(conf):
        """Map a [0, 1] confidence to a qualitative hedge, or None if unknown."""
        if conf is None:
            return None
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            return None
        if conf >= 0.8:
            return "fairly confident"
        if conf >= 0.5:
            return "moderately sure"
        return "low confidence, a rough estimate"

    @staticmethod
    def _coerce_state(grounded_state):
        """Accept a dict or a JSON string; return a dict or None."""
        if not grounded_state:
            return None
        if isinstance(grounded_state, str):
            try:
                grounded_state = json.loads(grounded_state)
            except (ValueError, TypeError):
                return None
        return grounded_state if isinstance(grounded_state, dict) else None

    def _format_grounded_block(self, grounded_state):
        """Render ``grounded_state`` as a soft first-person observation block.

        Returns ``""`` for an empty/malformed record so callers fall back to the
        base reasoner. Distances use the pipeline's calibrated metric scale; the
        camera-forward axis (z) is reported as depth/"away".
        """
        gs = self._coerce_state(grounded_state)
        if not gs:
            return ""
        lines = []
        objects = gs.get("objects") or []
        if objects:
            lines.append("Things I can make out in the scene:")
            for o in objects:
                label = o.get("label") or "an object"
                center = o.get("center_m")
                pos = ""
                if isinstance(center, (list, tuple)) and len(center) == 3:
                    pos = f" about {abs(center[2]):.2f} m away"
                conf = self._confidence_phrase(o.get("depth_conf"))
                hedge = f" (I'm {conf} about its distance)" if conf else ""
                lines.append(f"- {label}{pos}{hedge}")
        pairs = gs.get("pairs") or []
        if pairs:
            lines.append("Relative placements I've worked out:")
            for p in pairs:
                a = p.get("a") or "one object"
                b = p.get("b") or "another object"
                d = p.get("distance_m")
                dist = f" roughly {float(d):.2f} m apart" if d is not None else ""
                ir = p.get("inlier_ratio")
                rel = ""
                if ir is not None:
                    try:
                        ir = float(ir)
                        rel = f" (cross-view match is {'solid' if ir >= 0.7 else 'shaky'}, ~{ir:.2f})"
                    except (TypeError, ValueError):
                        rel = ""
                lines.append(f"- {a} and {b}{dist}{rel}")
        frames = gs.get("frames")
        try:
            if frames and int(frames) > 1:
                lines.append(f"(Pieced together from {int(frames)} views of the scene.)")
        except (TypeError, ValueError):
            pass
        return "\n".join(lines)

    def run(self, question: str, answer: str, image, grounded_state=None):
        """Generate a grounded, confidence-carrying trace.

        Falls back to :meth:`R1Reasoner.run` when no usable grounded state is
        supplied, preserving the original behavior exactly.
        """
        grounded_block = self._format_grounded_block(grounded_state)
        if not grounded_block:
            return super().run(question, answer, image)

        base64_image = self.encode_image(image)
        qa_pair = f"Question: {question} Answer: {answer}"
        user_text = (
            self.SOFT_SYSTEM_PROMPT
            + (self.PERSPECTIVE_ADDENDUM if self._is_perspective_question(question) else "")
            + "\nWhat I observe about the scene (with how sure I am):\n"
            + grounded_block
            + "\n\n"
            + qa_pair
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                    ],
                }
            ],
        )
        return response.choices[0].message.content

    def apply_transform(self, examples, images=None, text=None):
        """Like :meth:`R1Reasoner.apply_transform` but threads the grounded-state
        column into :meth:`run` so each trace is conditioned on the scene state."""
        images = images or self.image_column
        text = text or self.text_column
        gs_col = self.grounded_state_column

        is_batched = isinstance(examples[images], list)

        if not is_batched:
            conversation = examples[text]
            image_data = examples[images]
            if isinstance(image_data, list) and len(image_data) > 0:
                image_data = image_data[0]
            grounded_state = examples.get(gs_col)

            question, ans = self._find_random_qa_pair(conversation)
            if question is None or ans is None:
                examples["input"] = None
                examples["output"] = None
                examples["reasoning"] = None
            else:
                examples["input"] = self._format_input_string(question)
                examples["output"] = self.run(question, ans, image_data, grounded_state=grounded_state)
                examples["reasoning"] = "on"
            return examples

        conversation_batch = examples[text]
        image_batch = examples[images]
        gs_batch = examples.get(gs_col)

        inputs, outputs, reasonings = [], [], []
        for idx, (conv, img) in enumerate(zip(conversation_batch, image_batch)):
            if isinstance(img, list) and len(img) > 0:
                img = img[0]
            grounded_state = gs_batch[idx] if isinstance(gs_batch, list) and idx < len(gs_batch) else None

            question, ans = self._find_random_qa_pair(conv)
            if question is None or ans is None:
                inputs.append(None)
                outputs.append(None)
                reasonings.append(None)
                continue

            inputs.append(self._format_input_string(question))
            outputs.append(self.run(question, ans, img, grounded_state=grounded_state))
            reasonings.append("on")

        examples["input"] = inputs
        examples["output"] = outputs
        examples["reasoning"] = reasonings
        return examples


class GroundedBBoxTraceReasoner(SoftTraceReasoner):
    """Phase-3 teacher producing SpatialThinker-style scene-graph-grounded,
    bbox-verifiable traces (Batra et al., "SpatialThinker", arXiv:2511.07403,
    CC-BY-4.0).

    Emits the four-tag structure the dense reward scores:
      - ``<observe>`` what is visible in the image;
      - ``<scene>`` a JSON subgraph — ``objects`` with ``id``/``label``/``bbox``
        (``[x1, y1, x2, y2]`` absolute pixels) and ``relations`` triplets;
      - ``<think>`` first-person reasoning that INHERITS SoftTraceReasoner's
        grounded hints and propagates calibrated confidence (hedging low-
        confidence measurements rather than committing);
      - ``<answer>`` the final answer.

    The grounded state (object metric positions + surfaced depth_conf /
    inlier_ratio) is passed as HINTS; the teacher still predicts the pixel boxes
    from the image, and those predictions are verifiable against the localize-
    stage ground truth via :func:`vqasynth.spatial_reward.dense_reward`
    (Stage-2 verifiable-reward RL). Drop-in for the r1_reasoning stage; inherits
    apply_transform (threads the grounded_state column), image size is inferred
    from the PIL image when not supplied.
    """

    SPATIAL_SYSTEM_PROMPT = (
        "You are an embodied AI reasoning about a scene you are directly observing. Respond in FOUR parts, each "
        "in its own tags. (1) <observe> </observe>: briefly note what you actually see. (2) <scene> </scene>: emit "
        "a JSON scene subgraph of ONLY the objects relevant to the question, exactly as "
        '{"objects": [{"id": 0, "label": "...", "bbox": [x1, y1, x2, y2]}, ...], '
        '"relations": [{"subject": 0, "predicate": "...", "object": 1}, ...]} — bounding boxes in ABSOLUTE PIXEL '
        "coordinates within the image bounds, ids referenced by the relations. (3) <think> </think>: reason as a "
        "first-person internal monologue; PROPAGATE YOUR CONFIDENCE — lean on measurements you are sure of and "
        "hedge or cross-check low-confidence ones rather than committing to them as if certain. (4) <answer> "
        "</answer>: the final answer. Do not refer to information as given or provided; reason as if directly "
        "observed.\n"
    )

    @staticmethod
    def _infer_image_size(image):
        """Return (width, height) for a PIL image, else None."""
        try:
            if isinstance(image, Image.Image):
                return image.size
        except Exception:
            pass
        return None

    def run(self, question: str, answer: str, image, grounded_state=None, image_size=None):
        """Generate a scene-graph-grounded, bbox-verifiable, confidence-carrying
        trace in the four-tag format."""
        base64_image = self.encode_image(image)
        if image_size is None:
            image_size = self._infer_image_size(image)
        size_line = (
            f"Image size: {int(image_size[0])} x {int(image_size[1])}\n"
            if image_size and len(image_size) == 2
            else ""
        )
        grounded_block = self._format_grounded_block(grounded_state)
        hint = (
            "\nWhat I observe about the scene (hints, with how sure I am — still predict the boxes yourself):\n"
            + grounded_block
            + "\n"
            if grounded_block
            else ""
        )
        qa_pair = f"Question: {question} Answer: {answer}"
        persp = self.PERSPECTIVE_ADDENDUM if self._is_perspective_question(question) else ""
        user_text = self.SPATIAL_SYSTEM_PROMPT + persp + size_line + hint + "\n" + qa_pair
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                    ],
                }
            ],
        )
        return response.choices[0].message.content
