from __future__ import annotations

import json
from pathlib import Path

import pytest

from runpod_comfyui_mcp.workflow import (
    TextToImageParams,
    WorkflowError,
    attach_images,
    build_text_to_image,
    describe_template,
    find_node,
    load_template,
    nodes,
    prepare_text_to_image,
    set_scalar,
)

TEMPLATE = Path(__file__).resolve().parents[1] / "workflows" / "krea2_turbo_t2i.json"


@pytest.fixture()
def template() -> dict:
    return load_template(TEMPLATE)


def _params(**overrides) -> TextToImageParams:
    values = {"prompt": "a red fox in fresh snow, golden hour"}
    values.update(overrides)
    return TextToImageParams(**values)


def test_bundled_template_loads_and_describes():
    report = describe_template(TEMPLATE)

    assert report["valid"] is True
    assert report["nodes"]["sampler"] == "53"
    assert report["nodes"]["positive_encode"] == "51"
    assert report["nodes"]["negative_encode"] is None
    assert report["nodes"]["latent"] == "52"
    assert report["nodes"]["save"] == "29"
    assert report["nodes"]["loras"] == ["99"]
    assert any("negative" in warning.lower() for warning in report["warnings"])


def test_prompt_is_written_into_the_positive_encode(template):
    prepared = build_text_to_image(TEMPLATE, _params())
    assert prepared.workflow["51"]["inputs"]["text"] == "a red fox in fresh snow, golden hour"
    # The template must not be mutated in place.
    assert template["51"]["inputs"]["text"] != "a red fox in fresh snow, golden hour"


def test_seed_follows_the_primitive_link(template):
    prepared = build_text_to_image(TEMPLATE, _params(seed=777))

    assert prepared.workflow["120"]["inputs"]["value"] == 777
    assert prepared.workflow["53"]["inputs"]["seed"] == ["120", 0]
    seed_patch = next(p for p in prepared.report["patches"] if p["parameter"] == "seed")
    assert seed_patch["target"] == "120 (PrimitiveInt).value"


def test_scalar_sampler_and_latent_inputs(template):
    prepared = build_text_to_image(
        TEMPLATE,
        _params(width=1024, height=1024, steps=12, cfg=2.5, sampler="dpmpp_2m", scheduler="karras", lora_strength=0.6),
    )

    assert prepared.workflow["52"]["inputs"]["width"] == 1024
    assert prepared.workflow["52"]["inputs"]["height"] == 1024
    assert prepared.workflow["53"]["inputs"]["steps"] == 12
    assert prepared.workflow["53"]["inputs"]["cfg"] == 2.5
    assert prepared.workflow["53"]["inputs"]["sampler_name"] == "dpmpp_2m"
    assert prepared.workflow["53"]["inputs"]["scheduler"] == "karras"
    assert prepared.workflow["99"]["inputs"]["strength_model"] == 0.6


def test_negative_prompt_is_ignored_with_a_warning(template):
    prepared = build_text_to_image(TEMPLATE, _params(negative_prompt="blurry"))

    patch = next(p for p in prepared.report["patches"] if p["parameter"] == "negative_prompt")
    assert patch["applied"] is False
    assert any("negative_prompt" in warning for warning in prepared.report["warnings"])


def test_filename_prefix_is_sanitized(template):
    prepared = build_text_to_image(TEMPLATE, _params(filename_prefix="../../etc/passwd"))
    prefix = prepared.workflow["29"]["inputs"]["filename_prefix"]
    assert prefix == "etc/passwd"
    assert ".." not in prefix


def test_empty_prompt_is_rejected(template):
    with pytest.raises(WorkflowError, match="prompt"):
        build_text_to_image(TEMPLATE, _params(prompt="   "))


def test_missing_save_node_is_reported(template):
    template.pop("29")
    with pytest.raises(WorkflowError, match="SaveImage"):
        prepare_text_to_image(template, _params())


def test_missing_sampler_is_reported(template):
    template.pop("53")
    with pytest.raises(WorkflowError, match="KSampler"):
        prepare_text_to_image(template, _params())


def test_ui_export_is_rejected(tmp_path):
    ui_export = tmp_path / "ui.json"
    ui_export.write_text(json.dumps({"nodes": [], "links": []}), encoding="utf-8")
    with pytest.raises(WorkflowError, match="Export \\(API\\)"):
        load_template(ui_export)


def test_invalid_json_is_reported(tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(WorkflowError, match="not valid JSON"):
        load_template(broken)


def test_find_node_falls_back_to_class_type(template):
    # Renumber the sampler: the preferred id no longer matches.
    template["999"] = template.pop("53")
    assert find_node(template, "53", "KSampler") == "999"


def test_set_scalar_reports_a_computed_link(template):
    template["53"]["inputs"]["seed"] = ["200", 0]
    template["200"] = {"class_type": "SomeRandomNode", "inputs": {}}

    applied, where, note = set_scalar(template, "53", "seed", 5)

    assert applied is False
    assert where == "53.seed"
    assert "cannot be overridden" in note


def test_nodes_ignores_malformed_entries():
    index = nodes({"1": {"class_type": "KSampler", "inputs": {}}, "2": "nope", "3": {"inputs": {}}})
    assert list(index) == ["1"]


def test_attach_images_maps_load_image_nodes(template, tmp_path):
    template["170"] = {
        "class_type": "LoadImage",
        "inputs": {"image": "example.png"},
        "_meta": {"title": "Load Image - 1"},
    }
    image = tmp_path / "ref.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)

    workflow, images = attach_images(template, [image])

    assert workflow["170"]["inputs"]["image"] == "input_170.png"
    assert images[0]["name"] == "input_170.png"
    assert images[0]["image"]


def test_attach_images_without_loaders_is_an_error(template, tmp_path):
    image = tmp_path / "ref.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(WorkflowError, match="LoadImage"):
        attach_images(template, [image])


def test_attach_images_rejects_missing_file(template):
    template["170"] = {"class_type": "LoadImage", "inputs": {"image": "x.png"}}
    with pytest.raises(WorkflowError, match="not found"):
        attach_images(template, [TEMPLATE.parent / "absent.png"])


def test_payload_shape_matches_the_worker_contract(template):
    prepared = build_text_to_image(TEMPLATE, _params())
    payload = prepared.payload(output_type="base64")

    assert set(payload) == {"input"}
    assert payload["input"]["output_type"] == "base64"
    assert payload["input"]["workflow"]["53"]["class_type"] == "KSampler"
    assert "images" not in payload["input"]
