from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import math
import re

app = FastAPI()

SAFE_MAX = 9007199254740991

INTERVENTIONS = [
    "prompt_only",
    "retrieval",
    "lora",
    "qlora",
]

CHOOSE_CODES = [
    "INVALID_INPUT",
    "UNAVAILABLE",
    "QUALITY_FLOOR",
    "FRESHNESS_REQUIRED",
    "LATENCY_LIMIT",
    "MEMORY_LIMIT",
    "DATA_LIMIT",
    "COST_LIMIT",
]

REPAIR_CODES = [
    "INVALID_TOKEN",
    "INVALID_PARAMETER",
    "CHAT_TEMPLATE_COUNT",
    "INFERENCE_MODE",
    "FULL_MODEL_ARTIFACT",
    "ADAPTER_FILE_SET",
    "INCOMPLETE_CHECKPOINT",
    "MUTABLE_BASE_REVISION",
    "LINEAGE_MISMATCH",
    "EFFECTIVE_BATCH_MISMATCH",
    "EVAL_LEAKAGE",
    "EVAL_DROPOUT_ACTIVE",
    "RESUME_DIVERGENCE",
]

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")

REQUIRED_POLICY = {
    "minQuality",
    "freshnessRequired",
    "maxLatencyMs",
    "maxMemoryMb",
    "maxLabeledExamples",
    "maxTotalCost",
    "horizonRequests",
}

REQUIRED_CANDIDATE = {
    "name",
    "available",
    "quality",
    "freshness",
    "latencyMs",
    "memoryMb",
    "labeledExamples",
    "oneTimeCost",
    "recurringCost",
}


def utf8_sort(values):
    return sorted(values, key=lambda x: x.encode("utf-8"))


def unique_utf8(values):
    return utf8_sort(set(values))


def safe_int(v):
    return (
        isinstance(v, int)
        and not isinstance(v, bool)
        and 0 <= v <= SAFE_MAX
    )


def positive_safe_int(v):
    return safe_int(v) and v > 0


def finite_number(v):
    return (
        isinstance(v, (int, float))
        and not isinstance(v, bool)
        and math.isfinite(float(v))
    )


def finite_nonnegative(v):
    return finite_number(v) and float(v) >= 0


def finite_probability(v):
    return finite_number(v) and 0 <= float(v) <= 1


def read_json_safely(request):
    return request.json()


# ---------------------------------------------------------
# CHOOSE
# ---------------------------------------------------------

async def choose(body):
    policy = body.get("policy")
    candidates = body.get("candidates")

    if not isinstance(policy, dict):
        return None

    if not isinstance(candidates, list) or len(candidates) != 4:
        return None

    if set(policy.keys()) != REQUIRED_POLICY:
        return None

    if not isinstance(policy["freshnessRequired"], bool):
        return None

    if not finite_probability(policy["minQuality"]):
        return None

    for k in (
        "maxLatencyMs",
        "maxMemoryMb",
        "maxTotalCost",
    ):
        if not finite_nonnegative(policy[k]):
            return None

    for k in (
        "maxLabeledExamples",
        "horizonRequests",
    ):
        if not safe_int(policy[k]):
            return None

    names = []

    for c in candidates:
        if not isinstance(c, dict):
            return None

        if set(c.keys()) != REQUIRED_CANDIDATE:
            return None

        names.append(c["name"])

    if set(names) != set(INTERVENTIONS):
        return None

    if len(names) != len(set(names)):
        return None

    min_quality = float(policy["minQuality"])
    freshness_required = policy["freshnessRequired"]
    max_latency = float(policy["maxLatencyMs"])
    max_memory = float(policy["maxMemoryMb"])
    max_labeled = policy["maxLabeledExamples"]
    max_cost = float(policy["maxTotalCost"])
    horizon = policy["horizonRequests"]

    by_name = {c["name"]: c for c in candidates}

    total_costs = {}
    reasons = {}
    eligible = []

    for name in INTERVENTIONS:
        c = by_name[name]
        codes = []

        valid = True

        if not isinstance(c["available"], bool):
            valid = False

        if not finite_probability(c["quality"]):
            valid = False

        if not isinstance(c["freshness"], bool):
            valid = False

        for k in (
            "latencyMs",
            "memoryMb",
            "oneTimeCost",
            "recurringCost",
        ):
            if not finite_nonnegative(c[k]):
                valid = False

        if not safe_int(c["labeledExamples"]):
            valid = False

        if not valid:
            codes.append("INVALID_INPUT")
            total_costs[name] = None
            reasons[name] = unique_utf8(codes)
            continue

        cost = (
            float(c["oneTimeCost"])
            + float(horizon) * float(c["recurringCost"])
        )

        if not math.isfinite(cost) or cost < 0:
            codes.append("INVALID_INPUT")
            total_costs[name] = None
            reasons[name] = unique_utf8(codes)
            continue

        total_costs[name] = round(cost, 12)

        if not c["available"]:
            codes.append("UNAVAILABLE")

        if float(c["quality"]) < min_quality:
            codes.append("QUALITY_FLOOR")

        if freshness_required and c["freshness"] is not True:
            codes.append("FRESHNESS_REQUIRED")

        if float(c["latencyMs"]) > max_latency:
            codes.append("LATENCY_LIMIT")

        if float(c["memoryMb"]) > max_memory:
            codes.append("MEMORY_LIMIT")

        if c["labeledExamples"] > max_labeled:
            codes.append("DATA_LIMIT")

        if cost > max_cost:
            codes.append("COST_LIMIT")

        reasons[name] = unique_utf8(codes)

        if not codes:
            eligible.append(name)

    return {
        "selected": eligible[0] if eligible else None,
        "eligible": eligible,
        "totalCosts": {
            name: total_costs[name]
            for name in INTERVENTIONS
        },
        "reasonCodes": {
            name: reasons[name]
            for name in INTERVENTIONS
        },
    }


# ---------------------------------------------------------
# REPAIR
# ---------------------------------------------------------

def validate_tokens(tokens):
    if not isinstance(tokens, list) or len(tokens) == 0:
        return False, []

    valid = True

    for token in tokens:
        if not isinstance(token, dict):
            valid = False
            break

        if set(token.keys()) != {
            "id",
            "role",
            "padding",
            "text",
        }:
            valid = False
            break

        if not safe_int(token["id"]):
            valid = False
            break

        if token["role"] not in {
            "system",
            "user",
            "assistant",
        }:
            valid = False
            break

        if not isinstance(token["padding"], bool):
            valid = False
            break

        if not isinstance(token["text"], str):
            valid = False
            break

    if not valid:
        return False, [-100] * len(tokens)

    labels = []

    for token in tokens:
        if (
            token["role"] == "assistant"
            and token["padding"] is False
        ):
            labels.append(token["id"])
        else:
            labels.append(-100)

    return True, labels


def validate_parameters(parameters, allowed_targets):
    """
    Returns:
      valid
      full_model_artifact
      trainable_names
      trainable_count
    """

    if not isinstance(parameters, list):
        return False, False, [], 0

    if not isinstance(allowed_targets, list):
        return False, False, [], 0

    # Allowed targets must be non-empty unique strings.
    if len(allowed_targets) == 0:
        return False, False, [], 0

    if any(
        not isinstance(x, str) or x == ""
        for x in allowed_targets
    ):
        return False, False, [], 0

    if len(set(allowed_targets)) != len(allowed_targets):
        return False, False, [], 0

    allowed = set(allowed_targets)

    names_seen = set()

    for p in parameters:
        if not isinstance(p, dict):
            return False, False, [], 0

        if set(p.keys()) != {
            "name",
            "target",
            "numel",
        }:
            return False, False, [], 0

        name = p["name"]
        target = p["target"]
        numel = p["numel"]

        if not isinstance(name, str) or name == "":
            return False, False, [], 0

        if not isinstance(target, str) or target == "":
            return False, False, [], 0

        if not positive_safe_int(numel):
            return False, False, [], 0

        if name in names_seen:
            return False, False, [], 0

        names_seen.add(name)

    # A parameter is trainable iff BOTH conditions hold:
    #   1. target is explicitly allowed
    #   2. parameter is a LoRA A/B weight
    trainable = []

    for p in parameters:
        is_lora_weight = (
            p["name"].endswith(".lora_A.weight")
            or p["name"].endswith(".lora_B.weight")
        )

        if p["target"] in allowed and is_lora_weight:
            trainable.append(p)

    # At least one actual LoRA parameter is mandatory.
    if not trainable:
        return False, False, [], 0

    # If a parameter for an allowed target is not a LoRA A/B
    # parameter, this represents an attempt to train the base
    # model rather than only PEFT adapters.
    full_model = any(
        p["target"] in allowed
        and not (
            p["name"].endswith(".lora_A.weight")
            or p["name"].endswith(".lora_B.weight")
        )
        for p in parameters
    )

    trainable.sort(
        key=lambda p: p["name"].encode("utf-8")
    )

    count = 0

    for p in trainable:
        n = p["numel"]

        # Safe integer accumulation.
        if count > SAFE_MAX - n:
            return False, full_model, [], 0

        count += n

    return (
        not full_model,
        full_model,
        [p["name"] for p in trainable],
        count,
    )


def validate_adapter_files(files):
    required = [
        "adapter_config.json",
        "adapter_model.safetensors",
    ]

    if not isinstance(files, list):
        return False, []

    # Exactly two entries.
    if len(files) != 2:
        return False, utf8_sort(
            [x for x in files if isinstance(x, str)]
        )

    if any(not isinstance(x, str) for x in files):
        return False, []

    # Exactly once each.
    if (
        files.count("adapter_config.json") != 1
        or files.count("adapter_model.safetensors") != 1
    ):
        return False, utf8_sort(files)

    ordered = utf8_sort(files)

    return True, ordered


def validate_lineage(body):
    base = body.get("baseRevision")
    dataset = body.get("datasetDigest")
    code = body.get("codeDigest")
    config = body.get("configDigest")
    expected = body.get("expectedDigests")

    base_ok = (
        isinstance(base, str)
        and HEX40.fullmatch(base) is not None
    )

    digests_ok = all(
        isinstance(x, str)
        and HEX64.fullmatch(x) is not None
        for x in (
            dataset,
            code,
            config,
        )
    )

    if not base_ok:
        return False, True

    if not digests_ok:
        return False, False

    # expectedDigests is an evidence object. If supplied, all
    # expected lineage values must agree with the request.
    if not isinstance(expected, dict):
        return False, False

    for key, actual in (
        ("datasetDigest", dataset),
        ("codeDigest", code),
        ("configDigest", config),
    ):
        if key not in expected:
            return False, False

        expected_value = expected[key]

        if (
            not isinstance(expected_value, str)
            or HEX64.fullmatch(expected_value) is None
            or expected_value != actual
        ):
            return False, False

    return True, False


def validate_batch(body):
    mb = body.get("microBatch")
    ga = body.get("gradientAccumulation")
    replicas = body.get("replicas")
    expected = body.get("expectedEffectiveBatch")

    if not all(
        positive_safe_int(x)
        for x in (mb, ga, replicas, expected)
    ):
        return False

    # Avoid unsafe multiplication before comparing.
    if mb > SAFE_MAX // ga:
        return False

    product = mb * ga

    if product > SAFE_MAX // replicas:
        return False

    product *= replicas

    return product == expected


def validate_eval(body):
    train = body.get("trainRowIds")
    evaluation = body.get("evalRowIds")

    valid_lists = (
        isinstance(train, list)
        and isinstance(evaluation, list)
        and len(train) > 0
        and len(evaluation) > 0
    )

    if not valid_lists:
        return False

    if any(
        not isinstance(x, str) or x == ""
        for x in train
    ):
        return False

    if any(
        not isinstance(x, str) or x == ""
        for x in evaluation
    ):
        return False

    if len(set(train)) != len(train):
        return False

    if len(set(evaluation)) != len(evaluation):
        return False

    return set(train).isdisjoint(set(evaluation))


def validate_resume(body):
    uninterrupted = body.get("uninterruptedWeights")
    resumed = body.get("resumedWeights")
    tolerance = body.get("resumeTolerance")

    if not isinstance(uninterrupted, list):
        return False

    if not isinstance(resumed, list):
        return False

    if len(uninterrupted) == 0:
        return False

    if len(uninterrupted) != len(resumed):
        return False

    if not finite_nonnegative(tolerance):
        return False

    for a, b in zip(uninterrupted, resumed):
        if not finite_number(a) or not finite_number(b):
            return False

        if abs(float(a) - float(b)) > float(tolerance):
            return False

    return True


def repair(body):
    reasons = []

    # Tokens
    token_ok, labels = validate_tokens(body.get("tokens"))

    if not token_ok:
        reasons.append("INVALID_TOKEN")

    # Template
    template_pass = (
        body.get("templateApplications") == 1
    )

    if not template_pass:
        reasons.append("CHAT_TEMPLATE_COUNT")

    # Inference
    inference_ok = (
        body.get("inferenceMode") is False
    )

    if not inference_ok:
        reasons.append("INFERENCE_MODE")

    # PEFT parameters
    parameter_ok, full_model, trainable_params, trainable_count = (
        validate_parameters(
            body.get("parameters"),
            body.get("allowedTargets"),
        )
    )

    if not parameter_ok:
        reasons.append("INVALID_PARAMETER")

    if full_model:
        reasons.append("FULL_MODEL_ARTIFACT")

    peft_config_pass = parameter_ok

    # Adapter files
    adapter_ok, adapter_files = validate_adapter_files(
        body.get("artifactFiles")
    )

    if not adapter_ok:
        reasons.append("ADAPTER_FILE_SET")

    # Checkpoint
    checkpoint = body.get("checkpoint")

    checkpoint_complete = (
        isinstance(checkpoint, dict)
        and all(
            k in checkpoint
            for k in (
                "model",
                "optimizer",
                "scheduler",
                "step",
                "rng",
                "dataPosition",
            )
        )
    )

    if not checkpoint_complete:
        reasons.append("INCOMPLETE_CHECKPOINT")

    # Lineage
    lineage_pass, mutable_base = validate_lineage(body)

    if mutable_base:
        reasons.append("MUTABLE_BASE_REVISION")
    elif not lineage_pass:
        reasons.append("LINEAGE_MISMATCH")

    # Effective batch
    batch_pass = validate_batch(body)

    if not batch_pass:
        reasons.append("EFFECTIVE_BATCH_MISMATCH")

    # Evaluation isolation
    eval_isolated = validate_eval(body)

    if not eval_isolated:
        reasons.append("EVAL_LEAKAGE")

    # Evaluation dropout
    dropout_ok = (
        body.get("dropoutActiveDuringEval") is False
    )

    if not dropout_ok:
        reasons.append("EVAL_DROPOUT_ACTIVE")

    evaluation_deterministic = (
        eval_isolated and dropout_ok
    )

    # Resume
    resume_pass = validate_resume(body)

    if not resume_pass:
        reasons.append("RESUME_DIVERGENCE")

    return {
        "labels": labels,
        "templatePass": template_pass,
        "trainableParams": trainable_params,
        "trainableCount": trainable_count,
        "peftConfigPass": peft_config_pass,
        "adapterFiles": adapter_files,
        "checkpointComplete": checkpoint_complete,
        "lineagePass": lineage_pass,
        "evalIsolated": eval_isolated,
        "evaluationDeterministic": evaluation_deterministic,
        "resumePass": resume_pass,
        "reasonCodes": unique_utf8(reasons),
    }


# ---------------------------------------------------------
# ENDPOINT
# ---------------------------------------------------------

@app.post("/adapt")
async def adapt(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    if not isinstance(body, dict):
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    operation = body.get("operation")

    if operation == "choose":
        result = await choose(body)

        if result is None:
            return JSONResponse(
                status_code=400,
                content={"error": "INVALID_INPUT"},
            )

        return JSONResponse(content=result)

    if operation == "repair":
        return JSONResponse(content=repair(body))

    return JSONResponse(
        status_code=400,
        content={"error": "INVALID_INPUT"},
    )
