"""Engine registry + discovery.

Built-ins register on import. Third-party engines installed from GitHub
auto-register via the ``voicestudio.engines`` entry-point group::

    # in the external package's pyproject.toml
    [project.entry-points."voicestudio.engines"]
    f5 = "my_xtts_pkg:F5Engine"
"""
from importlib.metadata import entry_points

from .base import VoiceEngine

_REGISTRY: dict[str, type[VoiceEngine]] = {}


def register(cls: type[VoiceEngine]) -> type[VoiceEngine]:
    _REGISTRY[cls.name] = cls
    return cls


def _discover_external() -> None:
    try:
        eps = entry_points(group="voicestudio.engines")
    except TypeError:  # Python < 3.10 fallback
        eps = entry_points().get("voicestudio.engines", [])
    for ep in eps:
        try:
            cls = ep.load()
            # Duck-typed: GitHub engines don't need to import VoiceEngine,
            # just expose name/label/synthesize (see README).
            if (isinstance(getattr(cls, "name", None), str)
                    and callable(getattr(cls, "synthesize", None))
                    and cls.name not in _REGISTRY):
                _REGISTRY[cls.name] = cls
        except Exception as e:  # never break startup for a bad plugin
            print(f"[engines] skipping {ep.name}: {e}")


def _discover_builtin() -> None:
    from . import xtts as _xtts  # noqa: F401  (registers itself)

    _ = _xtts


_discover_builtin()
_discover_external()


def list_engines() -> dict[str, type[VoiceEngine]]:
    return dict(_REGISTRY)


def get_engine(name: str) -> VoiceEngine:
    try:
        return _REGISTRY[name]()
    except KeyError:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown engine {name!r}. Available: {available}")
