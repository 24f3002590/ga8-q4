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

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def utf8_sort(values):
    return sorted(set(values), key=lambda x: x.encode("utf-8"))


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


def probability(v):
    return finite_number(v) and 0 <= float(v) <= 1


# ============================================================
# CHOOSE
# ============================================================

def choose(body):
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

    if not probability(policy["minQuality"]):
        return None

    for key in (
        "maxLatencyMs",
        "maxMemoryMb",
        "maxTotalCost",
    ):
        if not finite_nonnegative(policy[key]):
            return None

    for key in (
        "maxLabeledExamples",
        "horizonRequests",
    ):
        if not safe_int(policy[key]):
            return None

    names = []

    for candidate in candidates:
        if not isinstance(candidate, dict):
            return None

        if set(candidate.keys()) != REQUIRED_CANDIDATE:
            return None

        names.append(candidate["name"])

    if set(names) != set(INTERVENTIONS):
        return None

    if len(set(names)) != 4:
        return None

    by_name = {c["name"]: c for c in candidates}

    min_quality = float(policy["minQuality"])
    freshness_required = policy["freshnessRequired"]
    max_latency = float(policy["maxLatencyMs"])
    max_memory = float(policy["maxMemoryMb"])
    max_labeled = policy["maxLabeledExamples"]
    max_total_cost = float(policy["maxTotalCost"])
    horizon = policy["horizonRequests"]

    total_costs = {}
    reason_codes = {}
    eligible = []

    for name in INTERVENTIONS:
        c = by_name[name]
        codes = []

        valid = True

        if not isinstance(c["available"], bool):
            valid = False

        if not probability(c["quality"]):
            valid = False

        if not isinstance(c["freshness"], bool):
            valid = False

        for key in (
            "latencyMs",
            "memoryMb",
            "oneTimeCost",
            "recurringCost",
        ):
            if not finite_nonnegative(c[key]):
                valid = False

        if not safe_int(c["labeledExamples"]):
            valid = False

        if not valid:
            codes.append("INVALID_INPUT")
            total_costs[name] = None
            reason_codes[name] = utf8_sort(codes)
            continue

        total = (
            float(c["oneTimeCost"])
            + float(horizon) * float(c["recurringCost"])
        )

        if not math.isfinite(total):
            codes.append("INVALID_INPUT")
            total_costs[name] = None
            reason_codes[name] = utf8_sort(codes)
            continue

        total_costs[name] = round(total, 12)

        if c["available"] is not True:
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

        if total > max_total_cost:
            codes.append("COST_LIMIT")

        codes = utf8_sort(codes)
        reason_codes[name] = codes

        if len(codes) == 0:
            eligible.append(name)

    return {
        "selected": eligible[0] if eligible else None,
        "eligible": eligible,
        "totalCosts": {
            name: total_costs[name]
            for name in INTERVENTIONS
        },
        "reasonCodes": {
            name: reason_codes[name]
            for name in INTERVENTIONS
        },
    }


# ============================================================
# TOKENS
# ============================================================

def validate_tokens(tokens):
    if not isinstance(tokens, list) or len(tokens) == 0:
        return False, []

    for token in tokens:
        if not isinstance(token, dict):
            return False, [-100] * len(tokens)

        if set(token.keys()) != {
            "id",
            "role",
            "padding",
            "text",
        }:
            return False, [-100] * len(tokens)

        if not safe_int(token["id"]):
            return False, [-100] * len(tokens)

        if token["role"] not in {
            "system",
            "user",
            "assistant",
        }:
            return False, [-100] * len(tokens)

        if not isinstance(token["padding"], bool):
            return False, [-100] * len(tokens)

        if not isinstance(token["text"], str):
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


# ============================================================
# PEFT PARAMETERS
# ============================================================

def validate_parameters(parameters, allowed_targets):
    """
    A parameter is trainable iff:
      - target is one of allowedTargets
      - name ends in .lora_A.weight or .lora_B.weight

    Other valid parameters may exist and are frozen.
    """

    if not isinstance(parameters, list):
        return False, [], 0

    if not isinstance(allowed_targets, list):
        return False, [], 0

    if len(allowed_targets) == 0:
        return False, [], 0

    if any(
        not isinstance(target, str) or target == ""
        for target in allowed_targets
    ):
        return False, [], 0

    if len(set(allowed_targets)) != len(allowed_targets):
        return False, [], 0

    allowed = set(allowed_targets)
    seen_names = set()

    for parameter in parameters:
        if not isinstance(parameter, dict):
            return False, [], 0

        if set(parameter.keys()) != {
            "name",
            "target",
            "numel",
        }:
            return False, [], 0

        name = parameter["name"]
        target = parameter["target"]
        numel = parameter["numel"]

        if not isinstance(name, str) or name == "":
            return False, [], 0

        if not isinstance(target, str) or target == "":
            return False, [], 0

        if not positive_safe_int(numel):
            return False, [], 0

        if name in seen_names:
            return False, [], 0

        seen_names.add(name)

    trainable = []

    for parameter in parameters:
        name = parameter["name"]
        target = parameter["target"]

        lora_weight = (
            name.endswith(".lora_A.weight")
            or name.endswith(".lora_B.weight")
        )

        if target in allowed and lora_weight:
            trainable.append(parameter)

    if not trainable:
        return False, [], 0

    trainable.sort(
        key=lambda p: p["name"].encode("utf-8")
    )

    count = 0

    for parameter in trainable:
        numel = parameter["numel"]

        if count > SAFE_MAX - numel:
            return False, [], 0

        count += numel

    return (
        True,
        [p["name"] for p in trainable],
        count,
    )


# ============================================================
# ADAPTER FILES
# ============================================================

def validate_adapter_files(files):
    required = {
        "adapter_config.json",
        "adapter_model.safetensors",
    }

    if not isinstance(files, list):
        return False, []

    if len(files) != 2:
        return False, []

    if any(not isinstance(x, str) for x in files):
        return False, []

    # Exactly once each; no extras or duplicates.
    if set(files) != required:
        return False, []

    if files.count("adapter_config.json") != 1:
        return False, []

    if files.count("adapter_model.safetensors") != 1:
        return False, []

    return True, sorted(
        files,
        key=lambda x: x.encode("utf-8"),
    )


# ============================================================
# CHECKPOINT
# ============================================================

def validate_checkpoint(checkpoint):
    if not isinstance(checkpoint, dict):
        return False

    required = {
        "model",
        "optimizer",
        "scheduler",
        "step",
        "rng",
        "dataPosition",
    }

    return required.issubset(set(checkpoint.keys()))


# ============================================================
# LINEAGE
# ============================================================

def validate_lineage(body):
    base_revision = body.get("baseRevision")
    dataset_digest = body.get("datasetDigest")
    code_digest = body.get("codeDigest")
    config_digest = body.get("configDigest")
    expected = body.get("expectedDigests")

    base_ok = (
        isinstance(base_revision, str)
        and HEX40.fullmatch(base_revision) is not None
    )

    if not base_ok:
        return False, True

    digest_values = (
        dataset_digest,
        code_digest,
        config_digest,
    )

    digests_ok = all(
        isinstance(value, str)
        and HEX64.fullmatch(value) is not None
        for value in digest_values
    )

    if not digests_ok:
        return False, False

    if not isinstance(expected, dict):
        return False, False

    for key, actual in (
        ("datasetDigest", dataset_digest),
        ("codeDigest", code_digest),
        ("configDigest", config_digest),
    ):
        if key not in expected:
            return False, False

        expected_value = expected[key]

        if not isinstance(expected_value, str):
            return False, False

        if HEX64.fullmatch(expected_value) is None:
            return False, False

        if expected_value != actual:
            return False, False

    return True, False


# ============================================================
# EFFECTIVE BATCH
# ============================================================

def validate_effective_batch(body):
    micro_batch = body.get("microBatch")
    accumulation = body.get("gradientAccumulation")
    replicas = body.get("replicas")
    expected = body.get("expectedEffectiveBatch")

    if not all(
        positive_safe_int(x)
        for x in (
            micro_batch,
            accumulation,
            replicas,
            expected,
        )
    ):
        return False

    if micro_batch > SAFE_MAX // accumulation:
        return False

    product = micro_batch * accumulation

    if product > SAFE_MAX // replicas:
        return False

    product *= replicas

    return product == expected


# ============================================================
# EVALUATION ISOLATION
# ============================================================

def validate_eval_isolation(body):
    train_ids = body.get("trainRowIds")
    eval_ids = body.get("evalRowIds")

    if not isinstance(train_ids, list):
        return False

    if not isinstance(eval_ids, list):
        return False

    if len(train_ids) == 0 or len(eval_ids) == 0:
        return False

    if any(
        not isinstance(x, str) or x == ""
        for x in train_ids
    ):
        return False

    if any(
        not isinstance(x, str) or x == ""
        for x in eval_ids
    ):
        return False

    if len(set(train_ids)) != len(train_ids):
        return False

    if len(set(eval_ids)) != len(eval_ids):
        return False

    return set(train_ids).isdisjoint(set(eval_ids))


# ============================================================
# RESUME
# ============================================================

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
        if not finite_number(a):
            return False

        if not finite_number(b):
            return False

        if abs(float(a) - float(b)) > float(tolerance):
            return False

    return True


# ============================================================
# REPAIR
# ============================================================

def repair(body):
    reasons = []

    # Tokenization / loss labels
    token_pass, labels = validate_tokens(
        body.get("tokens")
    )

    if not token_pass:
        reasons.append("INVALID_TOKEN")

    # Exactly one chat template application
    template_pass = (
        body.get("templateApplications") == 1
    )

    if not template_pass:
        reasons.append("CHAT_TEMPLATE_COUNT")

    # Must be training mode
    inference_pass = (
        body.get("inferenceMode") is False
    )

    if not inference_pass:
        reasons.append("INFERENCE_MODE")

    # PEFT parameters
    parameter_pass, trainable_params, trainable_count = (
        validate_parameters(
            body.get("parameters"),
            body.get("allowedTargets"),
        )
    )

    if not parameter_pass:
        reasons.append("INVALID_PARAMETER")

    # The supplied parameter inventory is considered a PEFT
    # configuration when valid and contains actual LoRA params.
    peft_config_pass = parameter_pass

    # Adapter artifacts
    adapter_pass, adapter_files = validate_adapter_files(
        body.get("artifactFiles")
    )

    if not adapter_pass:
        reasons.append("ADAPTER_FILE_SET")

    # Checkpoint
    checkpoint_complete = validate_checkpoint(
        body.get("checkpoint")
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
    effective_batch_pass = validate_effective_batch(body)

    if not effective_batch_pass:
        reasons.append("EFFECTIVE_BATCH_MISMATCH")

    # Train/eval row isolation
    eval_isolated = validate_eval_isolation(body)

    if not eval_isolated:
        reasons.append("EVAL_LEAKAGE")

    # Dropout must be inactive during evaluation
    dropout_ok = (
        body.get("dropoutActiveDuringEval") is False
    )

    if not dropout_ok:
        reasons.append("EVAL_DROPOUT_ACTIVE")

    evaluation_deterministic = (
        eval_isolated and dropout_ok
    )

    # Resume equivalence
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
        "reasonCodes": utf8_sort(reasons),
    }


# ============================================================
# ENDPOINT
# ============================================================

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
        result = choose(body)

        if result is None:
            return JSONResponse(
                status_code=400,
                content={"error": "INVALID_INPUT"},
            )

        return JSONResponse(content=result)

    if operation == "repair":
        return JSONResponse(
            content=repair(body)
        )

    return JSONResponse(
        status_code=400,
        content={"error": "INVALID_INPUT"},
    )
