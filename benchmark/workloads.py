import random

from benchmark.models import RequestSpec


def fixed_workload(
    total_requests: int,
    prompt_tokens: int,
    output_tokens: int,
) -> list[RequestSpec]:

    return [
        RequestSpec(
            request_id=i,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
        )
        for i in range(total_requests)
    ]


def mixed_workload(
    total_requests: int,
    seed: int = 42,
) -> list[RequestSpec]:

    random_generator = random.Random(seed)

    prompt_choices = [128, 512, 2048, 4096]
    output_choices = [32, 64, 128, 256]

    requests = []

    for request_id in range(total_requests):

        requests.append(
            RequestSpec(
                request_id=request_id,
                prompt_tokens=random_generator.choice(prompt_choices),
                output_tokens=random_generator.choice(output_choices),
                priority=random_generator.choice([0, 1, 2]),
            )
        )

    return requests