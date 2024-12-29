import os
from abc import ABC, abstractmethod
from typing import Optional, Any, Callable


class BasePipeline(ABC):
    def __init__(self, event_name: Optional[str] = None, logging_func: Callable = print,
                 cache_folder: str = "Data/temp"):
        self.is_chain = False
        self.name = event_name or self.__class__.__name__
        self.log = logging_func
        os.makedirs(cache_folder, exist_ok=True)
        self.cache_folder = cache_folder

    @abstractmethod
    def step(self, x: Any) -> Any:
        pass

    def __rshift__(self, other: "BasePipeline") -> "BasePipeline":
        """Chains the current pipeline step with another using the `>>` operator.

        Returns:
            A new pipeline that sequentially applies the `step` method of
            the current pipeline and the `step` method of the provided pipeline.
        """
        if not isinstance(other, BasePipeline):
            raise TypeError("Can only chain with another PandasPipeline instance.")

        return Chain(self, other)

    def __lshift__(self, other: "BasePipeline") -> "BasePipeline":
        """Chains the current pipeline step with another using the `<<` operator.

        Returns:
            A new pipeline that sequentially applies the `step` method of
            the provided pipeline and the `step` method of the current pipeline.
        """
        if not isinstance(other, BasePipeline):
            raise TypeError("Can only chain with another PandasPipeline instance.")

        return Chain(other, self)


class Chain(BasePipeline):
    """Handles a sequence of transformations from two PandasPipeline steps."""

    def __init__(self, first: BasePipeline, second: BasePipeline):
        super().__init__()
        self.is_chain = True
        self.events = []
        for candidate in [first, second]:
            if candidate.is_chain:
                self.events.extend(candidate.events)
            else:
                self.events.append(candidate)

    @property
    def sequence(self):
        return "Execution order: " + "->".join([event.name for (i, event) in enumerate(self.events)])

    def step(self, x: Any) -> Any:
        for i, event in enumerate(self.events):
            try:
                self.log(f"{i + 1}.Running {event.name}:")
                x = event.step(x)
            except Exception as e:
                raise RuntimeError(f"Error in pipeline step {event.name}: {str(e)}")
        return x


class PipeLambda(BasePipeline):
    def __init__(self, func: Callable, name: Optional[str]):
        """Initialize with a callable function that will be used in the pipeline."""
        super().__init__(event_name = name)
        if not callable(func):
            raise TypeError("The provided function is not callable.")
        self.f = func

    def step(self, x: Any) -> Any:
        """Applies the given function to the input data."""
        return self.f(x)
