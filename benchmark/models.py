from dataclasses import dataclass, field


@dataclass
class RequestSpec:
    request_id: int
    prompt_tokens: int
    output_tokens: int
    priority: int = 0


@dataclass
class RequestResult:
    request_id: int
    prompt_tokens: int
    requested_output_tokens: int

    start_time: float
    first_token_time: float | None
    end_time: float

    token_times: list[float] = field(default_factory=list)

    status_code: int | None = None
    error: str | None = None

    @property
    def output_tokens(self) -> int:
        return len(self.token_times)

    @property
    def ttft(self) -> float | None:
        if self.first_token_time is None:
            return None

        return self.first_token_time - self.start_time

    @property
    def e2e_latency(self) -> float:
        return self.end_time - self.start_time

    @property
    def itls(self) -> list[float]:
        if len(self.token_times) < 2:
            return []

        return [
            self.token_times[i] - self.token_times[i - 1]
            for i in range(1, len(self.token_times))
        ]

    @property
    def tpot(self) -> float | None:
        if len(self.token_times) < 2:
            return None

        return (
            self.token_times[-1] - self.token_times[0]
        ) / (len(self.token_times) - 1)

    @property
    def tokens_per_second(self) -> float | None:
        if self.tpot is None:
            return None

        return 1 / self.tpot