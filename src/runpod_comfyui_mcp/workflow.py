"""Turn a ComfyUI API-format template into a Runpod job payload.

This is the part worth understanding before you adapt the template to another
model. The file in ``workflows/`` is a *graph*: every node has an id, a
``class_type`` and an ``inputs`` map whose values are either literals or links of
the form ``["<node id>", <output slot>]``.

Because node ids are not stable across re-saved workflows (ComfyUI renumbers,
users delete and re-add nodes), every lookup here prefers a known id and then
falls back to a ``class_type`` search. Missing nodes produce a readable error
instead of a ``KeyError`` halfway through patching.
"""

from __future__ import annotations

import base64
import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
MAX_INPUT_IMAGES = 6

PRIMITIVE_TYPES = (
    "PrimitiveInt",
    "PrimitiveFloat",
    "PrimitiveString",
    "PrimitiveStringMultiline",
    "PrimitiveBoolean",
)

SAMPLER_TYPES = ("KSampler", "KSamplerAdvanced", "SamplerCustom", "KSampler (Efficient)")
LATENT_TYPES = ("EmptyLatentImage", "EmptySD3LatentImage", "EmptyLatentImageX", "EmptyLatentImageSDXL")
SAVE_TYPES = ("SaveImage", "Image Save", "SaveImageWebsocket")
LORA_TYPES = ("LoraLoaderModelOnly", "LoraLoader")


class WorkflowError(RuntimeError):
    """The template cannot be turned into a valid job."""


@dataclass(frozen=True)
class NodeRef:
    id: str
    class_type: str
    inputs: Mapping[str, Any] = field(default_factory=dict)
    title: str = ""


@dataclass
class Patch:
    """One parameter change, reported back to the caller."""

    parameter: str
    target: str
    applied: bool
    note: str = ""

    def as_dict(self) -> dict:
        data = {"parameter": self.parameter, "target": self.target, "applied": self.applied}
        if self.note:
            data["note"] = self.note
        return data


@dataclass
class PreparedWorkflow:
    """A patched workflow plus a human-readable record of what changed."""

    workflow: dict
    report: dict

    def payload(self, *, output_type: str = "base64", images: list[dict] | None = None) -> dict:
        """Build the JSON body expected by the Runpod ComfyUI worker."""
        body: dict[str, Any] = {"workflow": self.workflow, "output_type": output_type}
        if images:
            body["images"] = images
        return {"input": body}


# ── template loading ─────────────────────────────────────────────────────────
def load_template(path: str | Path) -> dict:
    """Load a ComfyUI *API* workflow, with a helpful error for UI exports."""
    template_path = Path(path)
    if not template_path.is_file():
        raise WorkflowError(
            f"Workflow template not found: {template_path}. "
            "Set WORKFLOW_PATH to a ComfyUI API-format JSON file."
        )

    try:
        data = json.loads(template_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WorkflowError(
            f"{template_path.name} is not valid JSON ({exc.msg} at line {exc.lineno})."
        ) from exc

    if not isinstance(data, dict) or not data:
        raise WorkflowError(
            f"{template_path.name} is not a ComfyUI API workflow "
            "(expected a JSON object mapping node ids to nodes)."
        )

    if "nodes" in data and "links" in data:
        raise WorkflowError(
            f"{template_path.name} is a ComfyUI *UI* export. Open it in ComfyUI and use "
            "Workflow → Export (API) — the UI format cannot be sent to a serverless worker."
        )

    return data


def nodes(workflow: Mapping[str, Any]) -> dict[str, NodeRef]:
    """Index the valid node entries of a workflow by id."""
    result: dict[str, NodeRef] = {}
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type")
        if not isinstance(class_type, str):
            continue
        inputs = node.get("inputs")
        meta = node.get("_meta")
        title = str(meta.get("title") or "") if isinstance(meta, dict) else ""
        result[str(node_id)] = NodeRef(
            id=str(node_id),
            class_type=class_type,
            inputs=inputs if isinstance(inputs, dict) else {},
            title=title,
        )
    return result


# ── graph helpers ────────────────────────────────────────────────────────────
def find_node(workflow: Mapping[str, Any], preferred: str | None, *class_types: str) -> str | None:
    """Return ``preferred`` when it matches, else the first node of ``class_types``."""
    index = nodes(workflow)
    if preferred is not None:
        candidate = index.get(str(preferred))
        if candidate and (not class_types or candidate.class_type in class_types):
            return candidate.id

    for ref in index.values():
        if ref.class_type in class_types:
            return ref.id
    return None


def _link(value: Any) -> tuple[str, int] | None:
    """Recognise a ComfyUI link value: ``["12", 0]``."""
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[0], (str, int))
        and isinstance(value[1], int)
    ):
        return str(value[0]), int(value[1])
    return None


def linked_node(workflow: Mapping[str, Any], node_id: str | None, key: str) -> str | None:
    """The node id feeding ``node_id.inputs[key]``, if it is a link."""
    if not node_id:
        return None
    ref = nodes(workflow).get(str(node_id))
    if ref is None:
        return None
    target = _link(ref.inputs.get(key))
    if target is None or target[0] not in workflow:
        return None
    return target[0]


def set_input(workflow: dict, node_id: str | None, key: str, value: Any) -> bool:
    """Set one input, creating the ``inputs`` map if the node lacks one."""
    if not node_id:
        return False
    node = workflow.get(str(node_id))
    if not isinstance(node, dict):
        return False
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        inputs = {}
        node["inputs"] = inputs
    inputs[key] = value
    return True


def _upstream_of_class(
    workflow: Mapping[str, Any],
    start: str | None,
    class_types: Iterable[str],
    max_depth: int = 5,
) -> str | None:
    """Walk upstream from ``start`` looking for one of ``class_types``.

    Follows the first connected input of each node, which is enough to step
    through thin conditioning wrappers such as ``ConditioningZeroOut``.
    """
    wanted = tuple(class_types)
    index = nodes(workflow)
    seen: set[str] = set()
    current = start

    for _ in range(max_depth):
        if not current or current in seen:
            return None
        seen.add(current)

        ref = index.get(current)
        if ref is None:
            return None
        if ref.class_type in wanted:
            return ref.id

        nxt: str | None = None
        for key in ("conditioning", "conditioning_1", "conditioning_2", "positive", "negative", "samples", "model", "clip", "vae"):
            nxt = linked_node(workflow, ref.id, key)
            if nxt:
                break
        if nxt is None:
            for value in ref.inputs.values():
                link = _link(value)
                if link and link[0] in workflow:
                    nxt = link[0]
                    break
        current = nxt

    return None


def set_scalar(workflow: dict, node_id: str | None, key: str, value: Any) -> tuple[bool, str, str]:
    """Write a scalar input, following a link to a primitive node when wired.

    Samplers commonly get their seed from a ``PrimitiveInt`` node rather than a
    literal, so writing ``KSampler.inputs.seed`` alone would be silently ignored.
    Returns ``(applied, target, note)``.
    """
    if not node_id:
        return False, "<missing>", f"no node to write {key} to"

    ref = nodes(workflow).get(str(node_id))
    if ref is None:
        return False, str(node_id), f"node {node_id} does not exist"

    link = _link(ref.inputs.get(key))
    if link is not None:
        upstream = nodes(workflow).get(link[0])
        if upstream and upstream.class_type in PRIMITIVE_TYPES:
            set_input(workflow, upstream.id, "value", value)
            return True, f"{upstream.id} ({upstream.class_type}).value", ""
        return (
            False,
            f"{ref.id}.{key}",
            f"{key} is driven by node {link[0]} ({upstream.class_type if upstream else 'unknown'}) "
            "and cannot be overridden",
        )

    set_input(workflow, ref.id, key, value)
    return True, f"{ref.id}.{key}", ""


def _set_text(workflow: dict, node_id: str, text: str) -> str:
    """Write a prompt into whichever text input this encoder uses."""
    ref = nodes(workflow).get(node_id)
    if ref is None:
        return "<missing>"

    if "text" in ref.inputs:
        keys = ["text"]
    elif "text_g" in ref.inputs or "text_l" in ref.inputs:
        keys = [key for key in ("text_g", "text_l") if key in ref.inputs]
    elif "prompt" in ref.inputs:
        keys = ["prompt"]
    else:
        keys = ["text"]

    for key in keys:
        set_input(workflow, node_id, key, text)
    return ", ".join(f"{node_id}.{key}" for key in keys)


def _sanitize_prefix(value: str) -> str:
    """Keep a filename prefix writable on the worker (no traversal, no drive letters)."""
    cleaned = re.sub(r"[^A-Za-z0-9 _./-]", "", str(value)).strip(" ./")
    cleaned = cleaned.replace("\\", "/")
    while "//" in cleaned:
        cleaned = cleaned.replace("//", "/")
    return cleaned[:80] or "runpod/image"


# ── text-to-image ────────────────────────────────────────────────────────────
@dataclass
class TextToImageParams:
    prompt: str
    negative_prompt: str | None = None
    width: int | None = None
    height: int | None = None
    batch_size: int | None = None
    steps: int | None = None
    cfg: float | None = None
    sampler: str | None = None
    scheduler: str | None = None
    denoise: float | None = None
    seed: int | None = None
    lora_strength: float | None = None
    filename_prefix: str | None = None


def _require_prompt(params: TextToImageParams) -> str:
    prompt = (params.prompt or "").strip()
    if not prompt:
        raise WorkflowError("`prompt` is required — describe the image you want.")
    return prompt


def prepare_text_to_image(template: Mapping[str, Any], params: TextToImageParams) -> PreparedWorkflow:
    """Patch a text-to-image graph and describe every change."""
    prompt = _require_prompt(params)
    workflow = copy.deepcopy(dict(template))
    patches: list[Patch] = []
    warnings: list[str] = []

    sampler_id = find_node(workflow, "53", *SAMPLER_TYPES)
    if not sampler_id:
        raise WorkflowError(
            "Workflow has no KSampler node, so the prompt, seed and step count cannot be set. "
            f"Nodes present: {', '.join(sorted({ref.class_type for ref in nodes(workflow).values()}))}"
        )

    positive_start = linked_node(workflow, sampler_id, "positive")
    positive_id = _upstream_of_class(workflow, positive_start, ("CLIPTextEncode",)) or find_node(
        workflow, "51", "CLIPTextEncode"
    )
    if not positive_id:
        raise WorkflowError(
            "Workflow has no CLIPTextEncode node feeding the sampler's `positive` input."
        )
    target = _set_text(workflow, positive_id, prompt)
    patches.append(Patch("prompt", target, True))

    # The negative input is frequently a ConditioningZeroOut wrapper around the
    # same encode, in which case a separate negative prompt is not possible.
    negative_start = linked_node(workflow, sampler_id, "negative")
    negative_id = _upstream_of_class(workflow, negative_start, ("CLIPTextEncode",))
    negative_usable = negative_id is not None and negative_id != positive_id

    if params.negative_prompt is not None and params.negative_prompt.strip():
        if negative_usable:
            _set_text(workflow, negative_id, params.negative_prompt.strip())
            patches.append(Patch("negative_prompt", negative_id, True))
        else:
            patches.append(Patch("negative_prompt", "-", False, "workflow reuses the positive encode"))
            warnings.append(
                "This workflow zeroes out or reuses the negative conditioning, so "
                "`negative_prompt` was ignored. Add a separate CLIPTextEncode node wired to "
                "the sampler's `negative` input to enable it."
            )

    latent_id = _upstream_of_class(
        workflow, linked_node(workflow, sampler_id, "latent_image"), LATENT_TYPES
    ) or find_node(workflow, "52", *LATENT_TYPES)

    if latent_id:
        latent = nodes(workflow)[latent_id]
        for name, value in (
            ("width", params.width),
            ("height", params.height),
            ("batch_size", params.batch_size),
        ):
            if value is None:
                continue
            if name in latent.inputs:
                set_input(workflow, latent_id, name, int(value))
                patches.append(Patch(name, f"{latent_id}.{name}", True))
            else:
                patches.append(Patch(name, "-", False, "input not present"))
                warnings.append(f"Node {latent_id} ({latent.class_type}) has no `{name}` input; ignored.")
    elif any(value is not None for value in (params.width, params.height, params.batch_size)):
        warnings.append(
            "Workflow has no EmptyLatentImage node; width/height/batch_size were ignored."
        )

    sampler = nodes(workflow)[sampler_id]
    scalar_inputs = (
        ("steps", params.steps, int),
        ("cfg", params.cfg, float),
        ("sampler_name", params.sampler, str),
        ("scheduler", params.scheduler, str),
        ("denoise", params.denoise, float),
    )
    for name, value, caster in scalar_inputs:
        if value is None:
            continue
        if name not in sampler.inputs:
            patches.append(Patch(name, "-", False, "input not present"))
            warnings.append(f"Sampler {sampler_id} ({sampler.class_type}) has no `{name}` input; ignored.")
            continue
        applied, where, note = set_scalar(workflow, sampler_id, name, caster(value))
        patches.append(Patch(name, where, applied, note))
        if note:
            warnings.append(f"`{name}`: {note}.")

    if params.seed is not None:
        applied, where, note = set_scalar(workflow, sampler_id, "seed", int(params.seed))
        patches.append(Patch("seed", where, applied, note))
        if note:
            warnings.append(f"`seed`: {note}.")

    if params.lora_strength is not None:
        lora_ids = [ref.id for ref in nodes(workflow).values() if ref.class_type in LORA_TYPES]
        if not lora_ids:
            patches.append(Patch("lora_strength", "-", False, "no LoRA nodes"))
            warnings.append("Workflow has no LoRA nodes; `lora_strength` was ignored.")
        for lora_id in lora_ids:
            set_input(workflow, lora_id, "strength_model", float(params.lora_strength))
            patches.append(Patch("lora_strength", f"{lora_id}.strength_model", True))

    save_id = find_node(workflow, "29", *SAVE_TYPES)
    if not save_id:
        raise WorkflowError(
            "Workflow has no SaveImage node; the worker would run and return nothing. "
            "Add a SaveImage node to the graph."
        )
    if params.filename_prefix:
        prefix = _sanitize_prefix(params.filename_prefix)
        set_input(workflow, save_id, "filename_prefix", prefix)
        patches.append(Patch("filename_prefix", f"{save_id}.filename_prefix", True))
    else:
        patches.append(Patch("filename_prefix", f"{save_id}.filename_prefix", False, "template default kept"))

    report = {
        "nodes": {
            "sampler": sampler_id,
            "positive_encode": positive_id,
            "negative_encode": negative_id if negative_usable else None,
            "latent": latent_id,
            "save": save_id,
        },
        "patches": [patch.as_dict() for patch in patches],
        "warnings": warnings,
    }
    return PreparedWorkflow(workflow=workflow, report=report)


def build_text_to_image(template_path: str | Path, params: TextToImageParams) -> PreparedWorkflow:
    """Load ``template_path`` and prepare it for :class:`TextToImageParams`."""
    template = load_template(template_path)
    prepared = prepare_text_to_image(template, params)
    prepared.report["template"] = str(Path(template_path).resolve())
    return prepared


# ── optional reference images (for img2img / later workflows) ────────────────
def _loader_order(ref: NodeRef) -> tuple[int, int]:
    """Order LoadImage nodes the way a human reads them: "Load Image - 2" first."""
    suffix = ref.title.rsplit("-", 1)[-1].strip() if ref.title else ""
    if suffix.isdigit():
        return (0, int(suffix))
    return (1, int(ref.id) if ref.id.isdigit() else 0)


def attach_images(
    workflow: dict,
    image_paths: Iterable[str | Path],
    *,
    max_images: int = MAX_INPUT_IMAGES,
) -> tuple[dict, list[dict]]:
    """Wire local image files into ``LoadImage`` nodes as worker placeholders.

    The serverless worker has no access to this machine, so bytes travel inside
    the job payload and each ``LoadImage`` node is pointed at a placeholder name
    that matches an entry in the worker's ``images`` array.
    """
    paths = [Path(path) for path in image_paths]
    if len(paths) > max_images:
        raise WorkflowError(f"At most {max_images} reference images are supported; got {len(paths)}.")

    index = nodes(workflow)
    loaders = sorted((ref for ref in index.values() if ref.class_type == "LoadImage"), key=_loader_order)

    if paths and not loaders:
        raise WorkflowError(
            "This workflow has no LoadImage node, so reference images cannot be attached. "
            "Use a text-to-image template or add a LoadImage node."
        )
    if len(paths) > len(loaders):
        raise WorkflowError(
            f"Workflow has {len(loaders)} LoadImage node(s) but {len(paths)} image(s) were supplied."
        )

    images: list[dict] = []
    for ref, path in zip(loaders, paths):
        if not path.is_file():
            raise WorkflowError(f"Reference image not found: {path}")
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise WorkflowError(
                f"{path.name} is not a supported image "
                f"({', '.join(sorted(IMAGE_EXTENSIONS))})."
            )
        name = f"input_{ref.id}{path.suffix.lower()}"
        images.append(
            {
                "name": name,
                "image": base64.b64encode(path.read_bytes()).decode("ascii"),
            }
        )
        set_input(workflow, ref.id, "image", name)

    return workflow, images


# ── diagnostics ──────────────────────────────────────────────────────────────
def describe_template(path: str | Path) -> dict:
    """Explain how a template maps onto the tool parameters (no GPU time)."""
    report: dict[str, Any] = {"path": str(Path(path).resolve()), "exists": Path(path).is_file()}
    try:
        workflow = load_template(path)
    except WorkflowError as exc:
        report.update({"valid": False, "error": str(exc)})
        return report

    index = nodes(workflow)
    sampler_id = find_node(workflow, "53", *SAMPLER_TYPES)
    positive_id = (
        _upstream_of_class(workflow, linked_node(workflow, sampler_id, "positive"), ("CLIPTextEncode",))
        if sampler_id
        else find_node(workflow, "51", "CLIPTextEncode")
    )
    negative_id = (
        _upstream_of_class(workflow, linked_node(workflow, sampler_id, "negative"), ("CLIPTextEncode",))
        if sampler_id
        else None
    )
    latent_id = (
        _upstream_of_class(workflow, linked_node(workflow, sampler_id, "latent_image"), LATENT_TYPES)
        if sampler_id
        else find_node(workflow, "52", *LATENT_TYPES)
    )
    save_id = find_node(workflow, "29", *SAVE_TYPES)
    lora_ids = [ref.id for ref in index.values() if ref.class_type in LORA_TYPES]

    problems: list[str] = []
    if not sampler_id:
        problems.append("No sampler node (KSampler).")
    if not positive_id:
        problems.append("No CLIPTextEncode node for the positive prompt.")
    if not save_id:
        problems.append("No SaveImage node, so the worker would return no images.")

    report.update(
        {
            "valid": not problems,
            "problems": problems,
            "node_count": len(index),
            "nodes": {
                "sampler": sampler_id,
                "positive_encode": positive_id,
                "negative_encode": negative_id if negative_id != positive_id else None,
                "latent": latent_id,
                "save": save_id,
                "loras": lora_ids,
            },
            "classes": sorted({ref.class_type for ref in index.values()}),
            "warnings": (
                []
                if negative_id != positive_id
                else ["Negative conditioning reuses the positive encode; `negative_prompt` is unavailable."]
            ),
        }
    )
    return report
