from concurrent.futures import Future


class LayerWiseLoadCounter:
    """CPU completion counter compatible with KV pools' layer wait hook."""

    def __init__(
        self,
        num_layers: int,
        on_layer_ready=None,
        *,
        error_message: str = "Layer-wise KV load failed.",
    ):
        self.num_layers = num_layers
        self.on_layer_ready = on_layer_ready
        self.error_message = error_message
        self.producer_index = -1
        self.consumer_index = -1
        self.futures: dict[int, list[Future]] = {}

    def update_producer(self) -> int:
        self.producer_index += 1
        self.futures[self.producer_index] = [Future() for _ in range(self.num_layers)]
        return self.producer_index

    def set_consumer(self, index: int) -> None:
        self.consumer_index = index

    def complete(self, index: int, layer: int) -> None:
        future = self.futures[index][layer]
        if not future.done():
            future.set_result(None)

    def fail(self, index: int, error: BaseException) -> None:
        for future in self.futures.get(index, ()):
            if not future.done():
                future.set_exception(error)

    def wait_until(self, threshold: int) -> None:
        index = self.consumer_index
        futures = self.futures.get(index)
        if futures is None:
            return
        try:
            futures[threshold].result()
            if self.on_layer_ready is not None:
                self.on_layer_ready(index, threshold)
        except BaseException as error:
            raise RuntimeError(self.error_message) from error
        finally:
            if threshold == self.num_layers - 1:
                self.futures.pop(index, None)

    def reset(self) -> None:
        self.producer_index = -1
        self.consumer_index = -1
        self.futures.clear()
