# cost_analysis_optimized_commented.py
# Realistic, heavily optimized cost/profit model for your AI study assistant
# Assumes students upload ~110 unique files once (5 courses × ~22 files), with occasional additions.
# Applies deduplication, lazy image processing, embedding reuse, model routing, caching,
# token compression, batching/early-exit, and amortization over retention.

from dataclasses import dataclass
from typing import Dict


# === Pricing constants (can be updated if OpenAI pricing changes) ===
@dataclass
class Pricing:
    gpt4o_input_per_million: float = 2.50       # $ per 1M input tokens for GPT-4o
    gpt4o_output_per_million: float = 10.00     # $ per 1M output tokens
    gpt4o_vision_per_million: float = 10.00     # Vision/image token processing cost
    ada_embedding_per_million: float = 0.10      # Embedding cost (Ada)


# === Usage assumptions for your scenario ===
@dataclass
class UsageAssumptions:
    # Files: 5 courses × ~22 files = ~110 unique uploads per user total lifetime
    avg_files_uploaded_total: int = 110
    images_per_file: int = 2                    # average images in each document
    tokens_per_file_text: int = 2000            # token size of text in each file
    tokens_per_image: int = 500                 # token equivalent for image analysis

    # Q&A usage (monthly)
    avg_questions_per_day: int = 12             # heavy study usage
    conversation_context_tokens: int = 800       # prior messages/context overhead
    tokens_per_question_input: int = 1500        # question + retrieved context tokens
    tokens_per_question_output: int = 600        # response length

    # Pricing / revenue assumptions
    price_per_user_per_month: float = 20.0       # what student pays

    # Infrastructure
    supabase_base: float = 25                    # base Supabase cost
    supabase_per_user: float = 0.10              # incremental per-user
    supabase_cap: float = 100                    # cap on Supabase cost
    storage_per_user: float = 0.05               # file storage per user per month

    # Retention horizon for amortizing one-time file costs (e.g., average student stays 6 months)
    retention_months: int = 6


# === Optimization parameters ===
@dataclass
class OptimizationSettings:
    # FILE-SIDE OPTS
    dedupe_fraction: float = 0.10                # 10% of file work saved by detecting duplicates
    lazy_image_fraction: float = 0.5             # only 50% of images processed proactively (others on demand)
    embedding_reuse_fraction: float = 0.90        # reuse/skip ~10% of embedding work (near-duplicate content)

    # QA-SIDE OPTS
    token_compression_fraction: float = 0.85      # reduce token volume by 15% via summarization/truncation
    model_routing_simple_fraction: float = 0.50   # 50% of questions are "simple" and can be cheaper
    simple_model_cost_multiplier: float = 0.5     # simple questions cost 50% of full model
    caching_hit_rate: float = 0.20                # 20% of repeated Q&A served from cache
    batching_and_early_exit_savings: float = 0.10  # additional 10% from batching / early exit heuristics


# === BASE COST CALCULATIONS ===

def base_file_processing_cost(pricing: Pricing, usage: UsageAssumptions) -> float:
    """
    Compute naive (non-optimized) per-user monthly file processing cost,
    then amortize over retention_months.
    Includes:
      - Text input token processing (GPT-4o input)
      - Image vision processing (GPT-4o vision)
      - Embedding (Ada)
    """
    text_input_cost = (
        usage.tokens_per_file_text
        * usage.avg_files_uploaded_total
        * pricing.gpt4o_input_per_million
        / 1_000_000
    )
    image_cost = (
        usage.tokens_per_image
        * usage.images_per_file
        * usage.avg_files_uploaded_total
        * pricing.gpt4o_vision_per_million
        / 1_000_000
    )
    embedding_cost = (
        usage.tokens_per_file_text
        * usage.avg_files_uploaded_total
        * pricing.ada_embedding_per_million
        / 1_000_000
    )

    total_raw = text_input_cost + image_cost + embedding_cost
    # Amortize one-time cost over retention window
    return total_raw / usage.retention_months


def base_qa_cost(pricing: Pricing, usage: UsageAssumptions) -> float:
    """
    Compute naive monthly question-answering cost per user.
    Includes:
      - Input tokens: question + context + conversation history
      - Output tokens: response
    """
    avg_questions_per_month = usage.avg_questions_per_day * 30
    input_tokens = usage.tokens_per_question_input + usage.conversation_context_tokens

    input_cost = (
        input_tokens * avg_questions_per_month * pricing.gpt4o_input_per_million / 1_000_000
    )
    output_cost = (
        usage.tokens_per_question_output * avg_questions_per_month * pricing.gpt4o_output_per_million / 1_000_000
    )
    return input_cost + output_cost


# === OPTIMIZED COSTS ===

def optimized_file_processing_cost(
    pricing: Pricing,
    usage: UsageAssumptions,
    optim: OptimizationSettings
) -> float:
    """
    Apply file-side optimizations:
      - Lazy image processing (only process a fraction upfront)
      - Embedding reuse (skip some embedding work)
      - Deduplication (skip redundant work)
      - Amortize over retention months
    """
    # Recompute components so we can apply selective reductions
    text_input_cost = (
        usage.tokens_per_file_text
        * usage.avg_files_uploaded_total
        * pricing.gpt4o_input_per_million
        / 1_000_000
    )
    # Only process lazy_image_fraction of images eagerly; others would be triggered on demand
    image_cost_eager = (
        usage.tokens_per_image
        * usage.images_per_file
        * usage.avg_files_uploaded_total
        * pricing.gpt4o_vision_per_million
        / 1_000_000
        * optim.lazy_image_fraction
    )
    # Embedding cost with reuse savings built in
    embedding_cost_effective = (
        usage.tokens_per_file_text
        * usage.avg_files_uploaded_total
        * pricing.ada_embedding_per_million
        / 1_000_000
        * optim.embedding_reuse_fraction
    )

    # Raw after applying lazy image and embedding reuse
    raw = text_input_cost + image_cost_eager + embedding_cost_effective

    # Deduplication reduces a fraction of that remaining work
    dedupe_savings = raw * optim.dedupe_fraction
    after_dedupe = raw - dedupe_savings

    # Amortize the one-time file cost
    return after_dedupe / usage.retention_months


def optimized_qa_cost(
    base_qa: float,
    usage: UsageAssumptions,
    optim: OptimizationSettings
) -> float:
    """
    Apply QA-side optimizations:
      - Token compression
      - Model routing (simple vs complex questions)
      - Caching (reduces effective volume)
      - Batching / early exit heuristics
    """
    qa = base_qa

    # 1. Token compression (summarize / truncate context)
    qa *= optim.token_compression_fraction

    # 2. Model routing: split into simple and complex questions
    simple_fraction = optim.model_routing_simple_fraction
    complex_fraction = 1 - simple_fraction

    # Simple questions use cheaper path
    simple_cost = simple_fraction * qa * optim.simple_model_cost_multiplier
    complex_cost = complex_fraction * qa  # full cost

    routed = simple_cost + complex_cost

    # 3. Caching: a portion of queries are answered from cache, reducing volume
    routed *= (1 - optim.caching_hit_rate)

    # 4. Batching / early-exit additional savings
    routed *= (1 - optim.batching_and_early_exit_savings)

    return routed


# === INFRASTRUCTURE ===

def infra_cost_per_user(num_users: int, usage: UsageAssumptions) -> float:
    """
    Supabase cost scales with user count but caps, plus per-user storage.
    Return per-user share of infrastructure.
    """
    supabase = min(usage.supabase_base + num_users * usage.supabase_per_user, usage.supabase_cap)
    storage = num_users * usage.storage_per_user
    total_infra = supabase + storage
    return total_infra / num_users if num_users > 0 else 0.0


# === AGGREGATION / SCALE SIMULATION ===

def calculate_for_scale(
    user_count: int,
    pricing: Pricing,
    usage: UsageAssumptions,
    optim: OptimizationSettings
) -> Dict:
    """
    Compute revenue, base costs, optimized costs, profits, and margins for a given user scale.
    """
    revenue = user_count * usage.price_per_user_per_month

    # Base per-user cost components
    file_base = base_file_processing_cost(pricing, usage)
    qa_base = base_qa_cost(pricing, usage)
    infra_per_user = infra_cost_per_user(user_count, usage)

    # Optimized per-user costs
    file_opt = optimized_file_processing_cost(pricing, usage, optim)
    qa_opt = optimized_qa_cost(qa_base, usage, optim)

    total_base_cost = (file_base + qa_base + infra_per_user) * user_count
    total_opt_cost = (file_opt + qa_opt + infra_per_user) * user_count

    profit_base = revenue - total_base_cost
    profit_opt = revenue - total_opt_cost

    margin_base = (profit_base / revenue) * 100 if revenue > 0 else 0
    margin_opt = (profit_opt / revenue) * 100 if revenue > 0 else 0

    return {
        "users": user_count,
        "revenue": revenue,
        "base": {
            "file_cost_per_user": file_base,
            "qa_cost_per_user": qa_base,
            "infra_per_user": infra_per_user,
            "total_cost": total_base_cost,
            "profit": profit_base,
            "margin_percent": margin_base
        },
        "optimized": {
            "file_cost_per_user": file_opt,
            "qa_cost_per_user": qa_opt,
            "infra_per_user": infra_per_user,
            "total_cost": total_opt_cost,
            "profit": profit_opt,
            "margin_percent": margin_opt
        },
        "breakdown": {
            "file_base": file_base,
            "qa_base": qa_base,
            "file_opt": file_opt,
            "qa_opt": qa_opt,
            "infra_per_user": infra_per_user
        }
    }


def run_analysis():
    """
    Driver: run across predefined user scales and print readable output.
    """
    pricing = Pricing()
    usage = UsageAssumptions()
    optim = OptimizationSettings()

    user_scales = [10, 50, 100, 200, 500]
    print("====== AI STUDY ASSISTANT COST & PROFIT ANALYSIS (WITH OPTIMIZATIONS) ======")
    for users in user_scales:
        res = calculate_for_scale(users, pricing, usage, optim)

        # Print base vs optimized comparison
        print(f"\n--- {users} users ---")
        print(f"Revenue: ${res['revenue']:,.2f}")
        print("Base (unoptimized) per-user costs:")
        print(f"  File processing: ${res['base']['file_cost_per_user']:.4f}")
        print(f"  Q&A:             ${res['base']['qa_cost_per_user']:.4f}")
        print(f"  Infra:           ${res['base']['infra_per_user']:.4f}")
        print(f"  Total cost/user: ${res['base']['file_cost_per_user'] + res['base']['qa_cost_per_user'] + res['base']['infra_per_user']:.4f}")
        print(f"  Total cost:      ${res['base']['total_cost']:,.2f}")
        print(f"  Profit:          ${res['base']['profit']:,.2f} ({res['base']['margin_percent']:.1f}%)")

        print("Optimized per-user costs:")
        print(f"  File processing: ${res['optimized']['file_cost_per_user']:.4f}")
        print(f"  Q&A:             ${res['optimized']['qa_cost_per_user']:.4f}")
        print(f"  Infra:           ${res['optimized']['infra_per_user']:.4f}")
        print(f"  Total cost/user: ${res['optimized']['file_cost_per_user'] + res['optimized']['qa_cost_per_user'] + res['optimized']['infra_per_user']:.4f}")
        print(f"  Total cost:      ${res['optimized']['total_cost']:,.2f}")
        print(f"  Profit:          ${res['optimized']['profit']:,.2f} ({res['optimized']['margin_percent']:.1f}%)")


if __name__ == "__main__":
    run_analysis()
