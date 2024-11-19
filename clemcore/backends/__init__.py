import abc
import importlib
import inspect
import json
import os
from types import SimpleNamespace
from dataclasses import dataclass
import nltk
from typing import Dict, List, Tuple, Any, Type, Union
import clemcore.utils.file_utils as file_utils


@dataclass(frozen=True)
class ModelSpec(SimpleNamespace):
    """Base class for model specifications.
    Holds all necessary information to make a model available for clembench: Responsible backend and any arbitrary data
    required by the backend. Also covers non-LLM 'models' like programmatic, slurk and direct user input.
    """
    PROGRAMMATIC_SPECS = ["mock", "dry_run", "programmatic", "custom", "_slurk_response"]
    HUMAN_SPECS = ["human", "terminal"]

    def __init__(self, **kwargs):
        """
        Args:
            kwargs: Keyword arguments used to set up the ModelSpec instance.
        """
        super().__init__(**kwargs)

    def unify(self, other: "ModelSpec") -> "ModelSpec":
        """Unify two ModelSpec instances.
        Args:
            other: The other ModelSpec instance this instance is to be unified with.
        Returns:
            The ModelSpec unification of this ModelSpec instance and the passed ModelSpec instance.
        Raises:
            ValueError: A ValueError exception is raised if the passed ModelSpec instance does not unify with this
                ModelSpec instance.
        """
        result = nltk.featstruct.unify(self.__dict__, other.__dict__)
        if result is None:
            raise ValueError(f"{self} does not unify with {other}")
        return ModelSpec(**result)

    def __repr__(self):
        """Get a string representation of this ModelSpec instance."""
        return f"ModelSpec({str(self)})"

    def __str__(self):
        """Get a string version of this ModelSpec instance."""
        return str(self.__dict__)

    def __getitem__(self, item):
        """Enable dict-like behavior."""
        return getattr(self, item)

    def __contains__(self, item):
        """Enable dict-like behavior."""
        return self.has_attr(item)

    def has_attr(self, attribute):
        """Check if this ModelSpec instance has the passed attribute.
        Args:
            attribute: The attribute to check for.
        """
        return hasattr(self, attribute)

    def has_temperature(self):
        """Check if this ModelSpec instance has a set 'temperature' attribute."""
        return self.has_attr("temperature")

    def has_backend(self):
        """Check if this ModelSpec instance has a set 'backend' attribute."""
        return self.has_attr("backend")

    @classmethod
    def from_name(cls, model_name: str):
        """Create a ModelSpec instance based on a model name.
        Args:
            model_name: The model name/ID as string.
        """
        if model_name is None:
            raise ValueError(f"Cannot create ModelSpec because model_name is None (but required)")
        return cls(model_name=model_name)

    @classmethod
    def from_dict(cls, spec: Dict):
        """Initialize a ModelSpec from a dictionary.
        Can be used to directly create a ModelSpec from a model registry entry dictionary.
        Args:
            spec: A model specification as dict.
        """
        return cls(**spec)

    def is_programmatic(self):
        """Check if this ModelSpec instance specifies a programmatic responder."""
        return self.model_name in ModelSpec.PROGRAMMATIC_SPECS

    def is_human(self):
        """Check if this ModelSpec instance specifies a human responder."""
        return self.model_name in ModelSpec.HUMAN_SPECS


# Load backend dynamically from "backends" sibling directory
# Note: The backends might use get_logger (circular import)
def load_credentials(backend, file_name="key.json") -> Dict:
    """Load login credentials and API keys from JSON file.
    Args:
        backend: Name of the backend/API provider to load key for.
        file_name: Name of the key file. Defaults to key.json in the clembench root directory.
    Returns:
        Dictionary with {backend: {api_key: key}}.
    """
    key_file = os.path.join(file_utils.project_root(), file_name)
    with open(key_file) as f:
        creds = json.load(f)
    assert backend in creds, f"No '{backend}' in {file_name}. See README."
    assert "api_key" in creds[backend], f"No 'api_key' in {file_name}. See README."
    return creds


class Model(abc.ABC):
    """A local/remote proxy for a model to be called."""
    def __init__(self, model_spec: ModelSpec):
        """
        Args:
            model_spec: A ModelSpec instance that specifies the model and the backend to be used.
        """
        assert hasattr(model_spec, "model_name"), "The passed ModelSpec must have a `model_name` attribute"
        self.model_spec = model_spec
        self.__gen_args = dict()

    def set_gen_args(self, **gen_args):
        """Set text generation inference parameters for this model.
        Currently implemented: Temperature and maximum number of tokens to generate.
        Args:
            gen_args: Keyword arguments/dict containing extra information needed for the generation process.
        """
        self.__gen_args = dict(gen_args)

    def set_gen_arg(self, arg_name, arg_value):
        """Set a text generation inference parameter for this model.
        Currently implemented: Temperature and maximum number of tokens to generate.
        Args:
            arg_name: The name of the generation inference parameter.
            arg_value: The value of the generation inference parameter.
        """
        self.__gen_args[arg_name] = arg_value

    def get_gen_arg(self, arg_name):
        """Get the value of a text generation inference parameter for this model.
        Currently implemented: Temperature and maximum number of tokens to generate.
        Args:
            arg_name: The name of the generation inference parameter.
        """
        assert arg_name in self.__gen_args, f"No '{arg_name}' in gen_args given but is expected"
        return self.__gen_args[arg_name]

    def get_temperature(self):
        """Get the value of the temperature text generation inference parameter for this model.
        Returns:
            The sampling temperature used for the generation process.
        """
        return self.get_gen_arg("temperature")

    def get_max_tokens(self):
        """Get the value of the maximum number of tokens text generation inference parameter for this model.
        Returns:
            The maximum number of tokens generated during the generation process.
        """
        return self.get_gen_arg("max_tokens")

    def get_name(self) -> str:
        """Get the name of this model.
        Returns:
            The name of the model as a string.
        """
        return self.model_spec.model_name

    def __repr__(self):
        """Get a string representation of this Model instance."""
        return str(self)

    def __str__(self):
        """Get the name of this Model instance's model.
        Returns:
            The name of the model as a string.
        """
        return self.get_name()

    def __eq__(self, other: "Model"):
        """Check if another assumed Model instance has the same model.
        Also checks if the passed object is a Model instance.
        Args:
            other: The other object to check for being a Model instance and having the same model name.
        Returns:
            False if either the passed object is not a Model instance or the passed object is a Model instance, but has
            a different model name; True if the passed object is both a Model instance and has the same model name.
        """
        if not isinstance(other, Model):
            return False
        return self.get_name() == other.get_name()

    @abc.abstractmethod
    def generate_response(self, messages: List[Dict]) -> Tuple[Any, Any, str]:
        """Put prompt in model-specific format and get its response.

        Args:
            messages (List[Dict]): The dialogue context represented as a list
                of turns. Entry element is a dictionary containing one key
                "role", whose value is either "user" or "assistant", and one
                key "content", whose value is the message as a string.

        Returns:
            Tuple[Any, Any, str]: The first element is the prompt object as
            passed to the LLM (i.e. after any model-specific manipulation).
            Return the full prompt object, not only the message string.

            The second element is the response object as gotten from the model,
            before any manipulation. Return the full prompt object, not only
            the message string.

            These must be returned just to be logged by the GM for later inspection.

            The third element is the response text, i.e. only the actual message
            generated by the model as a string (after any needed manipulation,
            like .strip() or excluding the input prompt).
        """
        pass


class Backend(abc.ABC):
    """Abstract base class for clembench backends.
    All clembench backend classes must be child classes of this base class."""
    @abc.abstractmethod
    def get_model_for(self, model_spec: ModelSpec) -> Model:
        """Get a Model instance for the model specific by ModelSpec.
        Must be implemented by every clembench backend.
        Args:
            model_spec: A ModelSpec instance specifying the model to return a corresponding Model child class instance
                for the appropriate backend.
        Returns:
            A Model instance using the appropriate backend.
        """
        pass

    def __repr__(self):
        """Get a string representation of this Backend instance."""
        return str(self)

    def __str__(self):
        """Get a string name of the class of this Backend child class instance."""
        return f"{self.__class__.__name__}"


class CustomResponseModel(Model):
    """Model child class to handle custom programmatic responses."""
    def __init__(self, model_spec=ModelSpec(model_name="programmatic")):
        super().__init__(model_spec)
        self.set_gen_args(temperature=0.0)  # dummy value for get_temperature()

    def generate_response(self, messages: List[Dict]) -> Tuple[Any, Any, str]:
        raise NotImplementedError("This should never be called but is handled in Player for now.")


class HumanModel(Model):
    """Model child class to handle human (terminal) responses."""
    def __init__(self, model_spec=ModelSpec(model_name="human")):
        super().__init__(model_spec)
        self.set_gen_args(temperature=0.0)  # dummy value for get_temperature()

    def generate_response(self, messages: List[Dict]) -> Tuple[Any, Any, str]:
        raise NotImplementedError("This should never be called but is handled in Player for now.")


def is_backend(obj):
    """Check if an object is a Backend child class (instance).
    Args:
        obj: The object to be checked.
    Returns:
        True if the object is a Backend child class (instance); False otherwise.
    """
    if inspect.isclass(obj) and issubclass(obj, Backend):
        return True
    return False


_backend_registry: Dict[str, Backend] = dict()  # we store references to the class constructor
_model_registry: List[ModelSpec] = list()  # we store model specs so that users might use model_name for lookup


def load_custom_model_registry(_model_registry_path: str = None, is_optional=True):
    """Load a custom model registry from file.
    Args:
        _model_registry_path: Path to the custom model registry file to be loaded. If not passed,
            model_registry_custom.json.template in the backends directory will be loaded.
        is_optional: If loading the custom model registry file is optional. Default: True.
    """
    if not _model_registry_path:
        _model_registry_path = os.path.join(file_utils.clemcore_root(), "backends",
                                            "model_registry_custom.json")
    load_model_registry(_model_registry_path, is_mandatory=not is_optional)


def load_model_registry(_model_registry_path: str = None, is_mandatory=True):
    """Load the model registry from file.
    Args:
        _model_registry_path: Path to the model registry file to be loaded. If not passed, model_registry.json in the
            backends directory will be loaded.
        is_mandatory: If loading the model registry file is mandatory. Default: True.
    Raises:
        FileNotFoundError: Will be raised if the model registry JSON file is not located at the passed
            or default path (backends/model_registry.json).
        ValueError: Will be raised if the model registry to be loaded contains faulty model entry.
    """
    if not _model_registry_path:
        _model_registry_path = os.path.join(file_utils.clemcore_root(), "backends", "model_registry.json")
    if not os.path.isfile(_model_registry_path):
        if is_mandatory:
            raise FileNotFoundError(f"The file model registry at '{_model_registry_path}' does not exist. "
                                    f"Create model registry as a model_registry.json file and try again.")
        else:
            return  # do nothing
    with open(_model_registry_path, encoding='utf-8') as f:
        _model_listing = json.load(f)
        for _model_entry in _model_listing:
            _model_spec: ModelSpec = ModelSpec.from_dict(_model_entry)
            if not _model_spec.has_backend():
                raise ValueError(
                    f"Missing backend definition in model spec '{_model_spec}'. "
                    f"Check or update the backends/model_registry.json and try again."
                    f"A minimal model spec is {{'model_id':<id>,'backend':<backend>}}.")
            _model_registry.append(_model_spec)


def _register_backend(backend_name: str):
    """Dynamically loads the Backend in the file with name <backend_name>_api.py into the _backend_registry.
    Raises an exception if no such file exists or the Backend class could not be found.
    Args:
        backend_name: The <backend_name> prefix of the <backend_name>_api.py file.
    Returns:
        The Backend subclass for the passed backend name.
    Raises:
        FileNotFoundError: Will be raised if no backend python file with the passed name can be found in the backends
            directory.
        LookupError: Will be raised if the backend python file with the passed name does not contain exactly one Backend
            subclass.
    """
    backends_root = os.path.join(file_utils.clemcore_root(), "backends")
    backend_module = f"{backend_name}_api"
    backend_path = os.path.join(backends_root, f"{backend_module}.py")
    if not os.path.isfile(backend_path):
        raise FileNotFoundError(f"The file '{backend_path}' does not exist. "
                                f"Create such a backend file or check the backend_name '{backend_name}'.")
    module = importlib.import_module(f"backends.{backend_module}")
    backend_subclasses = inspect.getmembers(module, predicate=is_backend)
    if len(backend_subclasses) == 0:
        raise LookupError(f"There is no Backend defined in {backend_module}. "
                          f"Create such a class and try again or check the backend_name '{backend_name}'.")
    if len(backend_subclasses) > 1:
        raise LookupError(f"There is more than one Backend defined in {backend_module}.")
    _, backend_cls = backend_subclasses[0]
    _backend_registry[backend_name] = backend_cls()
    return backend_cls


def _load_model_for(model_spec: ModelSpec) -> Model:
    """Load a model backend class based on the passed ModelSpec.
    Registers backend if it is not already registered.
    Args:
        model_spec: The ModelSpec specifying the model to load the backend class for.
    Returns:
        The Model subclass for the model specified in the passed ModelSpec.
    """
    backend_name = model_spec.backend
    if backend_name not in _backend_registry:
        _register_backend(backend_name)
    backend_cls = _backend_registry[backend_name]
    return backend_cls.get_model_for(model_spec)


def get_model_for(model_spec: Union[str, Dict, ModelSpec]) -> Model:
    """Get a Model subclass based on the passed specification.
    Args:
        model_spec: The model spec for which a supporting backend has to be found. Can be either a model name as string,
            a dictionary version of a model specification or a ModelSpec instance.
    Returns:
        The registered backend that supports the specified model.
    Raises:
        ValueError: Will be raised if the model specification does not contain fitting backend information - after
            unification with registered model specifications.
    """
    assert len(_model_registry) > 0, "Model registry is empty. Load a model registry and try again."

    if isinstance(model_spec, str):
        model_spec = ModelSpec.from_name(model_spec)
    if isinstance(model_spec, dict):
        model_spec = ModelSpec.from_dict(model_spec)

    if model_spec.is_human():
        return HumanModel(model_spec)
    if model_spec.is_programmatic():
        return CustomResponseModel(model_spec)

    for registered_spec in _model_registry:
        try:
            model_spec = model_spec.unify(registered_spec)
            break  # use first model spec that does unify (doesn't throw an error)
        except ValueError:
            continue

    if not model_spec.has_backend():
        raise ValueError(
            f"Model spec requires 'backend' after unification, but not found in model spec '{model_spec}'. "
            f"Check or update the backends/model_registry.json or pass the backend directly and try again. "
            f"A minimal model spec is {{'model_id':<id>,'backend':<backend>}}.")
    model = _load_model_for(model_spec)
    return model


class ContextExceededError(Exception):
    """Exception to be raised when the messages passed to a backend instance exceed the context limit of the model."""
    tokens_used: int = int()
    tokens_left: int = int()
    context_size: int = int()

    def __init__(self, info_str: str = "Context limit exceeded", tokens_used: int = 0,
                 tokens_left: int = 0, context_size: int = 0):
        """
        Args:
            info_str: String informing about context limit being exceeded. To optionally be modified with further
                information by the backend class eventually raising this error.
            tokens_used: The number of tokens used by the context that lead to this error being raised.
            tokens_left: The number of tokens left in the context limit. Will be negative if this error is raised,
                absolute value being the number of tokens that exceed the context limit.
            context_size: The size of the context/the context limit.
        """
        info = f"{info_str} {tokens_used}/{context_size}"
        super().__init__(info)
        self.tokens_used = tokens_used
        self.tokens_left = tokens_left
        self.context_size = context_size
