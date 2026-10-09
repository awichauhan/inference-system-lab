from benchmark.models import RequestResult


def test_request_metrics():

    result = RequestResult(
        request_id=1,
        prompt_tokens=128,
        requested_output_tokens=4,
        start_time=10.0,
        first_token_time=10.2,
        end_time=10.5,
        token_times=[
            10.2,
            10.3,
            10.4,
            10.5,
        ],
        status_code=200,
        error=None,
    )

    assert round(result.ttft, 3) == 0.2
    assert round(result.e2e_latency, 3) == 0.5
    assert round(result.tpot, 3) == 0.1

    assert [
        round(value, 3)
        for value in result.itls
    ] == [
        0.1,
        0.1,
        0.1,
    ]