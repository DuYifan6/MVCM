import run_strict_electra_baseline_seed42 as electra


def result_row(value=.8):
    return {
        "variant": "electra",
        "seed": 42,
        "checkpoint_epoch": 2,
        "val_loss": .2,
        "fake_threshold": .55,
        "fake_accuracy": value,
        "fake_macro_f1": value - .01,
        "ai_accuracy": value + .02,
        "ai_macro_f1": value + .01,
        "output_dir": "runs/electra",
    }


def test_electra_extension_is_one_fixed_dual_head_baseline():
    assert electra.VARIANT == "electra"
    assert electra.FAKE_THRESHOLD == .55
    assert electra.MODEL_SPEC["pretrained_model"] == "google/electra-base-discriminator"
    assert electra.MODEL_SPEC["lr"] == 2e-5
    assert electra.MODEL_SPEC["weight_decay"] == .01


def test_electra_summary_is_partial_safe():
    empty = electra.summarize(None, "validation_frozen")
    assert empty["planned"] == 1
    assert empty["completed"] == 0
    assert empty["by_variant"]["electra"] is None

    complete = electra.summarize(result_row(.9), "locked_test")
    assert complete["completed"] == 1
    assert complete["by_variant"]["electra"]["fake_accuracy"] == .9
    assert complete["fake_threshold"] == .55
