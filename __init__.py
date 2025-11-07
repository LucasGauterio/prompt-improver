try:
    from .improver import PromptImprover
    from .strategies import (
        RoleStrategy,
        FewShotStrategy,
        ChainOfThoughtStrategy,
        SelfConsistencyStrategy,
        TreeOfThoughtStrategy,
        SkeletonOfThoughtStrategy,
        ReActStrategy
    )
except ImportError:
    # Fallback for when running as a script
    from improver import PromptImprover
    from strategies import (
        RoleStrategy,
        FewShotStrategy,
        ChainOfThoughtStrategy,
        SelfConsistencyStrategy,
        TreeOfThoughtStrategy,
        SkeletonOfThoughtStrategy,
        ReActStrategy
    )

__all__ = [
    'PromptImprover',
    'RoleStrategy',
    'FewShotStrategy',
    'ChainOfThoughtStrategy',
    'SelfConsistencyStrategy',
    'TreeOfThoughtStrategy',
    'SkeletonOfThoughtStrategy',
    'ReActStrategy',
]

