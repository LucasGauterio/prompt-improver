try:
    from .base import BaseStrategy
    from .role import RoleStrategy
    from .few_shot import FewShotStrategy
    from .chain_of_thought import ChainOfThoughtStrategy
    from .self_consistency import SelfConsistencyStrategy
    from .tree_of_thought import TreeOfThoughtStrategy
    from .skeleton_of_thought import SkeletonOfThoughtStrategy
    from .react import ReActStrategy
except ImportError:
    # Fallback for when running as a script
    from strategies.base import BaseStrategy
    from strategies.role import RoleStrategy
    from strategies.few_shot import FewShotStrategy
    from strategies.chain_of_thought import ChainOfThoughtStrategy
    from strategies.self_consistency import SelfConsistencyStrategy
    from strategies.tree_of_thought import TreeOfThoughtStrategy
    from strategies.skeleton_of_thought import SkeletonOfThoughtStrategy
    from strategies.react import ReActStrategy

__all__ = [
    'BaseStrategy',
    'RoleStrategy',
    'FewShotStrategy',
    'ChainOfThoughtStrategy',
    'SelfConsistencyStrategy',
    'TreeOfThoughtStrategy',
    'SkeletonOfThoughtStrategy',
    'ReActStrategy',
]

