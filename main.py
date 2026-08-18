# main.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import math
import re

app = FastAPI()

INTERVENTIONS = ["prompt_only", "retrieval", "lora", "qlora"]

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

SAFE_INT_MAX = 9007199254740991

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


def utf8_sorted_unique(values):
    return sorted(set(values), key=lambda x: x.encode("utf-8"))


def is_safe_int(x):
    return (
        isinstance(x, int)
        and not isinstance(x, bool)
        and 0 <= x <= SAFE_INT_MAX
    )


def is_finite_number(x):
    return (
        isinstance(x, (int, float))
        and not isinstance(x, bool)
        and math.isfinite(float(x))
    )


def nonnegative_finite(x):
    return is_finite_number(x) and float(x) >= 0


def round12(x):
    return round(float(x), 12)


async def read_json(request: Request):
    try:
        body = await request.json()
    except Exception:
        return None
    return body


def choose(body):
    policy = body.get("policy")
    candidates = body.get("candidates")

    if not isinstance(policy, dict) or not isinstance(candidates, list):
        return None

    if set(policy.keys()) != REQUIRED_POLICY:
        return None

    if len(candidates) != 4:
        return None

    # Exactly one candidate for each intervention.
    names = [c.get("name") if isinstance(c, dict) else None for c in candidates]
    if sorted(names) != sorted(INTERVENTIONS):
        return None

    # Validate policy.
    if not isinstance(policy["freshnessRequired"], bool):
        return None

    if not nonnegative_finite(policy["minQuality"]):
        return None
    if float(policy["minQuality"]) > 1:
        return None

    for k in ("maxLatencyMs", "maxMemoryMb", "maxTotalCost"):
        if not nonnegative_finite(policy[k]):
            return None

    for k in ("maxLabeledExamples", "horizonRequests"):
        if not is_safe_int(policy[k]):
            return None

    min_quality = float(policy["minQuality"])
    freshness_required = policy["freshnessRequired"]
    max_latency = float(policy["maxLatencyMs"])
    max_memory = float(policy["maxMemoryMb"])
    max_labeled = policy["maxLabeledExamples"]
    max_cost = float(policy["maxTotalCost"])
    horizon = policy["horizonRequests"]

    total_costs = {}
    reason_codes = {}
    eligible = []

    for c in candidates:
        if not isinstance(c, dict):
            return None
        if set(c.keys()) != REQUIRED_CANDIDATE:
            return None

        name = c["name"]

        valid_candidate = True

        if not isinstance(c["available"], bool):
            valid_candidate = False

        if not is_finite_number(c["quality"]) or not 0 <= float(c["quality"]) <= 1:
            valid_candidate = False

        for k in ("latencyMs", "memoryMb", "oneTimeCost", "recurringCost"):
            if not nonnegative_finite(c[k]):
                valid_candidate = False

        if not is_safe_int(c["labeledExamples"]):
            valid_candidate = False

        codes = []

        if not valid_candidate:
            codes.append("INVALID_INPUT")

            # Cost is still required to exist in the response.
            total_costs[name] = None
            reason_codes[name] = utf8_sorted_unique(codes)
            continue

        total = round12(
            float(c["oneTimeCost"])
            + horizon * float(c["recurringCost"])
        )
        total_costs[name] = total

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

        if total > max_cost:
            codes.append("COST_LIMIT")

        codes = utf8_sorted_unique(codes)
        reason_codes[name] = codes

        if not codes:
            eligible.append(name)

    eligible.sort(key=lambda x: INTERVENTIONS.index(x))

    return {
        "selected": eligible[0] if eligible else None,
        "eligible": eligible,
        "totalCosts": {name: total_costs[name] for name in INTERVENTIONS},
        "reasonCodes": {
            name: reason_codes[name] for name in INTERVENTIONS
        },
    }


def repair(body):
    tokens = body.get("tokens")
    parameters = body.get("parameters")
    allowed_targets = body.get("allowedTargets")

    token_valid = True
    labels = []

    if not isinstance(tokens, list) or len(tokens) == 0:
        token_valid = False
    else:
        for tok in tokens:
            if not isinstance(tok, dict):
                token_valid = False
                break

            if set(tok.keys()) != {"id", "role", "padding", "text"}:
                token_valid = False
                break

            if not is_safe_int(tok["id"]):
                token_valid = False
                break

            if tok["role"] not in {"system", "user", "assistant"}:
                token_valid = False
                break

            if not isinstance(tok["padding"], bool):
                token_valid = False
                break

            if not isinstance(tok["text"], str):
                token_valid = False
                break

    if token_valid:
        labels = [
            tok["id"]
            if tok["role"] == "assistant" and tok["padding"] is False
            else -100
            for tok in tokens
        ]
    else:
        labels = [-100] * (len(tokens) if isinstance(tokens, list) else 0)

    template_pass = body.get("templateApplications") == 1
    inference_pass = body.get("inferenceMode") is False

    # Parameters
    parameter_valid = True
    trainable = []
    parameter_names = set()

    if not isinstance(parameters, list):
        parameter_valid = False
        parameters = []

    for p in parameters:
        if not isinstance(p, dict):
            parameter_valid = False
            continue

        if set(p.keys()) != {"name", "target", "numel"}:
            parameter_valid = False
            continue

        if (
            not isinstance(p["name"], str)
            or not p["name"]
            or not isinstance(p["target"], str)
            or not p["target"]
            or not is_safe_int(p["numel"])
            or p["numel"] <= 0
        ):
            parameter_valid = False
            continue

        if p["name"] in parameter_names:
            parameter_valid = False
        parameter_names.add(p["name"])

    if not isinstance(allowed_targets, list) or len(allowed_targets) == 0:
        parameter_valid = False
        allowed_targets = []
    elif (
        any(not isinstance(x, str) or not x for x in allowed_targets)
        or len(set(allowed_targets)) != len(allowed_targets)
    ):
        parameter_valid = False

    allowed_set = set(allowed_targets)

    if parameter_valid:
        trainable = [
            p for p in parameters
            if (
                p["target"] in allowed_set
                and (
                    p["name"].endswith(".lora_A.weight")
                    or p["name"].endswith(".lora_B.weight")
                )
            )
        ]

        if len(trainable) == 0:
            parameter_valid = False

    trainable.sort(key=lambda p: p["name"].encode("utf-8"))

    trainable_params = [p["name"] for p in trainable]
    trainable_count = (
        sum(p["numel"] for p in trainable)
        if parameter_valid
        else 0
    )

    # If a parameter targets an allowed PEFT target but is not a LoRA
    # parameter, treat it as full-model adaptation.
    full_model_artifact = False
    if parameter_valid:
        for p in parameters:
            if p["target"] in allowed_set and not (
                p["name"].endswith(".lora_A.weight")
                or p["name"].endswith(".lora_B.weight")
            ):
                full_model_artifact = True
                break

    peft_config_pass = parameter_valid and not full_model_artifact

    # Adapter files
    artifact_files = body.get("artifactFiles")
    expected_files = ["adapter_config.json", "adapter_model.safetensors"]

    adapter_pass = (
        isinstance(artifact_files, list)
        and len(artifact_files) == 2
        and all(isinstance(x, str) for x in artifact_files)
        and sorted(artifact_files, key=lambda x: x.encode("utf-8"))
        == sorted(expected_files, key=lambda x: x.encode("utf-8"))
    )

    adapter_files = (
        sorted(artifact_files, key=lambda x: x.encode("utf-8"))
        if isinstance(artifact_files, list)
        and all(isinstance(x, str) for x in artifact_files)
        else []
    )

    # Checkpoint
    checkpoint = body.get("checkpoint")
    checkpoint_complete = (
        isinstance(checkpoint, dict)
        and all(k in checkpoint for k in [
            "model",
            "optimizer",
            "scheduler",
            "step",
            "rng",
            "dataPosition",
        ])
    )

    # Lineage
    base_revision = body.get("baseRevision")
    dataset_digest = body.get("datasetDigest")
    code_digest = body.get("codeDigest")
    config_digest = body.get("configDigest")
    expected = body.get("expectedDigests")

    base_valid = isinstance(base_revision, str) and bool(HEX40.fullmatch(base_revision))
    digest_valid = all(
        isinstance(x, str) and bool(HEX64.fullmatch(x))
        for x in (dataset_digest, code_digest, config_digest)
    )

    lineage_pass = base_valid and digest_valid

    if isinstance(expected, dict):
        expected_values = {
            "datasetDigest": expected.get("datasetDigest"),
            "codeDigest": expected.get("codeDigest"),
            "configDigest": expected.get("configDigest"),
        }
        if any(
            expected_values[k] is not None
            and expected_values[k] != body.get(k)
            for k in expected_values
        ):
            lineage_pass = False

        # If expected digests are supplied, all three must match.
        if expected:
            for k in ("datasetDigest", "codeDigest", "configDigest"):
                if expected.get(k) != body.get(k):
                    lineage_pass = False
    else:
        lineage_pass = False

    # Batch
    mb = body.get("microBatch")
    ga = body.get("gradientAccumulation")
    replicas = body.get("replicas")
    expected_batch = body.get("expectedEffectiveBatch")

    batch_valid = all(
        is_safe_int(x) and x > 0
        for x in (mb, ga, replicas, expected_batch)
    )

    effective_batch_pass = (
        batch_valid and mb * ga * replicas == expected_batch
    )

    # Evaluation isolation
    train_ids = body.get("trainRowIds")
    eval_ids = body.get("evalRowIds")

    ids_valid = (
        isinstance(train_ids, list)
        and isinstance(eval_ids, list)
        and len(train_ids) > 0
        and len(eval_ids) > 0
        and all(isinstance(x, str) and x for x in train_ids)
        and all(isinstance(x, str) and x for x in eval_ids)
        and len(set(train_ids)) == len(train_ids)
        and len(set(eval_ids)) == len(eval_ids)
    )

    eval_isolated = ids_valid and set(train_ids).isdisjoint(set(eval_ids))

    dropout_active = body.get("dropoutActiveDuringEval")
    dropout_ok = dropout_active is False

    evaluation_deterministic = eval_isolated and dropout_ok

    # Resume
    uninterrupted = body.get("uninterruptedWeights")
    resumed = body.get("resumedWeights")
    tolerance = body.get("resumeTolerance")

    resume_valid = (
        isinstance(uninterrupted, list)
        and isinstance(resumed, list)
        and len(uninterrupted) > 0
        and len(uninterrupted) == len(resumed)
        and all(is_finite_number(x) for x in uninterrupted)
        and all(is_finite_number(x) for x in resumed)
        and is_finite_number(tolerance)
        and float(tolerance) >= 0
    )

    resume_pass = False
    if resume_valid:
        resume_pass = all(
            abs(float(a) - float(b)) <= float(tolerance)
            for a, b in zip(uninterrupted, resumed)
        )

    reasons = []

    if not token_valid:
        reasons.append("INVALID_TOKEN")

    if not parameter_valid:
        reasons.append("INVALID_PARAMETER")

    if body.get("templateApplications") != 1:
        reasons.append("CHAT_TEMPLATE_COUNT")

    if body.get("inferenceMode") is not False:
        reasons.append("INFERENCE_MODE")

    if full_model_artifact:
        reasons.append("FULL_MODEL_ARTIFACT")

    if not adapter_pass:
        reasons.append("ADAPTER_FILE_SET")

    if not checkpoint_complete:
        reasons.append("INCOMPLETE_CHECKPOINT")

    if not base_valid:
        reasons.append("MUTABLE_BASE_REVISION")

    if not lineage_pass and base_valid:
        reasons.append("LINEAGE_MISMATCH")

    if not effective_batch_pass:
        reasons.append("EFFECTIVE_BATCH_MISMATCH")

    if not eval_isolated:
        reasons.append("EVAL_LEAKAGE")

    if not dropout_ok:
        reasons.append("EVAL_DROPOUT_ACTIVE")

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
        "reasonCodes": utf8_sorted_unique(reasons),
    }


@app.post("/adapt")
async def adapt(request: Request):
    body = await read_json(request)

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
        return JSONResponse(content=repair(body))

    return JSONResponse(
        status_code=400,
        content={"error": "INVALID_INPUT"},
    )
