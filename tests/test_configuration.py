from citadel.infrastructure.resources import image_input
import hashlib
from dataclasses import replace

from citadel.application.prompts import PromptBuilder
from citadel.configuration import PromptBundle
from citadel.infrastructure.files import fingerprint


def test_extracted_prompts_preserve_review_and_quality_inputs(model_case):
    bundle = PromptBundle.load()
    expected = {
        "review_system": "9ff295e47873403ac67d61941b3f117e531851eeac10690f8129fba65e55ab17",
        "gripper": "ef156b57c07aaf313928a84483e98d61e6e6bc0c72c64c3230f8e01a5c00ae29",
        "quality": "07b467f2b13632e8aeb88150d4ab4f05de1436924bf721cd340a9e0c7920e7dd",
    }
    assert {key: hashlib.sha256(value.encode()).hexdigest()
            for key, value in bundle.as_dict().items()} == expected
    work, resources, profile, media = model_case
    builder = PromptBuilder(bundle, image_input)
    assert builder.review(work, resources, profile, media)[0]["content"] == bundle.review_system
    assert builder.quality(work, media)[0]["content"] == bundle.quality


def test_prompt_snapshot_is_stable_and_changes_signature(model_case):
    original = PromptBundle.load()
    changed = replace(original, review_system=original.review_system + "新的审核说明。")
    assert fingerprint(original.as_dict()) != fingerprint(changed.as_dict())
    work, resources, profile, media = model_case
    builder = PromptBuilder(original, image_input)
    assert builder.review(work, resources, profile, media)[0]["content"] == original.review_system
