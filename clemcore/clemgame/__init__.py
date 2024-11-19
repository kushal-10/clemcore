import abc
import collections
import copy
import json
import os.path
import sys
from datetime import datetime
from typing import List, Dict, Tuple, Any
from tqdm import tqdm
from types import SimpleNamespace
import importlib
import importlib.util
import inspect
import logging

import clemcore.backends as backends
import clemcore.utils.file_utils as file_utils
import clemcore.utils.transcript_utils as transcript_utils
import clemcore.clemgame.metrics as ms

logger = logging.getLogger(__name__)
stdout_logger = logging.getLogger("clemcore.run")

game_registry = []  # list of game specs to load from dynamically


class GameSpec(SimpleNamespace):
    """Base class for game specifications.
    Holds all necessary information to play game in clembench (see README for list of attributes)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # check for required fields
        if "game_name" not in self:
            raise KeyError(f"No game name specified in entry {kwargs}")
        if "game_path" not in self:
            raise KeyError(f"No game path specified in {kwargs}")
        # make game_path absolute
        if not os.path.isabs(self.game_path):
            self.game_path = os.path.join(file_utils.project_root(), self.game_path)

    def __repr__(self):
        """Returns string representation of this GameSpec."""
        return f"GameSpec({str(self)})"

    def __str__(self):
        """Returns GameSpec instance attribute dict as string."""
        return str(self.__dict__)

    def __getitem__(self, item):
        """Access GameSpec instance attributes like dict items.
        Args:
            item: The string name of the instance attribute to get.
        Returns:
            The value of the GameSpec instance attribute, or if the instance does not have the attribute, the string
            passed as argument to this method.
        """
        return getattr(self, item)

    def __contains__(self, attribute):
        """Check GameSpec instance attributes like dict keys.
        Args:
            attribute: The string name of the instance attribute to check for.
        Returns:
            True if the GameSpec instance contains an attribute with the passed string name, False otherwise.
        """
        return hasattr(self, attribute)

    @classmethod
    def from_dict(cls, spec: Dict):
        """Initialize a GameSpec from a dictionary.
        Can be used to directly create a GameSpec from a game registry entry.
        Args:
            spec: A game-specifying dict.
        Returns:
            A GameSpec instance with the data specified by the passed dict.
        """
        return cls(**spec)

    def matches(self, spec: Dict):
        """Check if the game features match a given specification.
        Args:
            spec: A game-specifying dict.
        Returns:
            True if the game features match the passed specification, False otherwise.
        Raises:
            KeyError: The GameSpec instance does not contain an attribute corresponding to a key in the passed
                game-specifying dict.
        """
        for key, value in spec.items():
            if not self.__contains__(key):
                raise KeyError(f"The specified key '{key}' for selecting games is not set in the game registry "
                               f"for game '{self['game_name']}'")
            if type(self[key]) == str:
                if not self[key] == value:
                    return False
            elif type(self[key]) == list:
                if value not in self[key]:
                    return False
        return True

    def get_game_file(self):
        """Get the file path of the master.py of the game specified by this GameSpec instance.
        Main game file must be called master.py in game directory.
        Returns:
            The file path of the master.py of the game specified by this GameSpec instance as a string.
        """
        return os.path.join(self.game_path, "master.py")

    def game_file_exists(self):
        """Check if master.py can be located at the specified game_path.
        Returns:
            True if the master.py is located at the specified game_path, False otherwise.
        """
        return True if os.path.isfile(self.get_game_file()) else False


def load_custom_game_registry(_game_registry_path: str = None, is_optional=True):
    """Load a custom game registry.
    Handled as module-level variable.
    Args:
        _game_registry_path: The path to a custom game registry JSON file. Optional: If not passed, default path is
            used.
        is_optional: Determines if a custom game registry is not required.
    """
    # optional custom registry loaded first, so that these entries come first in the game registry list
    if not _game_registry_path:
        _game_registry_path = os.path.join(file_utils.clemcore_root(), "clemgame", "game_registry_custom.json")
    load_game_registry(_game_registry_path, is_mandatory=not is_optional)


def load_game_registry(_game_registry_path: str = None, is_mandatory=True):
    """Load the game registry.
    Handled as module-level variable.
    Args:
        _game_registry_path: The path to the game registry JSON file. Optional: If not passed, default path is used.
        is_mandatory: If True, a FileNotFoundError is raised if the game registry JSON file does not exist at the
            path specified in _game_registry_path (or the default path, if nothing is passed to _game_registry_path).
    Raises:
        FileNotFoundError: If True is passed to is_mandatory, FileNotFoundError is raised if the game registry JSON file
            does not exist at the path specified in _game_registry_path (or the default path, if nothing is passed to
            _game_registry_path).
    """
    if not _game_registry_path:
        _game_registry_path = os.path.join(file_utils.clemcore_root(), "clemgame", "game_registry.json")
    if not os.path.isfile(_game_registry_path):
        if is_mandatory:
            raise FileNotFoundError(f"The file game registry at '{_game_registry_path}' does not exist. "
                                    f"Create game registry as a game_registry.json file and try again.")
        else:
            return  # do nothing
    with open(_game_registry_path, encoding='utf-8') as gr:
        _game_listing = json.load(gr)
        for _game_entry in _game_listing:
            _game_spec: GameSpec = GameSpec.from_dict(_game_entry)
            game_registry.append(_game_spec)


def select_game(game_name: str) -> GameSpec:
    """Select a GameSpec from the game registry by game name.
    Args:
        game_name: String name of the selected game.
    Returns:
        A GameSpec instance from the game registry corresponding to the passed game_name.
    Raises:
        ValueError: No game specification matching the passed game_name was found in the game registry.
    """
    # return first entry that matches game_name
    for game in game_registry:
        if game["game_name"] == game_name:
            if game.game_file_exists():
                return game
            else:
                raise ValueError(f"Game master file master.py not found in {game['game_path']}."
                               f"Update clemcore/clemgame/game_registry.json (or game_registry_custom.json) with the right path for {game_name}.")
    raise ValueError(f"No games found matching the given specification '{game_name}'. "
                          "Make sure the game name matches the name in clemcore/clemgame/game_registry.json (or game_registry_custom.json)")
    # extension to select subset of games
    # (postponed because it introduces more complexity
    # on things like how to specify specific episodes (which could, however be integrated into the game spec
    # and then selected through the custom game_spec for a specific run),
    # and thus can be easier done by looping over an
    # explicit list of games with a bash script (see clembench/scripts/run_benchmark.sh)

    # select relevant games from game registry
    # selected_games = []
    # properties = {}
    # is_single_game = True
    # if game_name.endswith(".json"):
    #     is_single_game = False
    #     with open(os.path.join(file_utils.project_root(), game_name)) as f:
    #         properties = json.load(f)
    #     # add default values
    #     if "lang" not in properties:
    #         properties["language"] = "en"
    #     if "image" not in properties:
    #         properties["image"] = "none"
    #     # examples:
    #     # {"benchmark" : "2.0"} # run all English textual games marked for benchmark version 2.0
    #     # {"benchmark" : "1.5", "lang": "ru"} # run all games of benchmark version 1.5 for which Russian versions exist
    #     # {"main_game": "matchit"} # to run all English textual matchit game versions
    #     # {"image": "single", "main_game": "matchit"} # to run all English multimodal matchit game versions
    #
    # if is_single_game:
    #     # return first entry that matches game_name
    #     for game in game_registry:
    #         if game["game_name"] == game_name:
    #             return game
    # else:
    #     for game in game_registry:
    #         if game.matches(properties):
    #             selected_games.append(game)
    #
    # if len(selected_games) == 0:
    #     raise ValueError(f"No games found matching the given specification '{game_name}'. "
    #                      "Make sure game name or attribute names and values match game_registry.json")
    # return selected_games


class Player(abc.ABC):
    """A participant of a game.

    A player can respond via a custom implementation, human input or a language model:

    - the programmatic players are called via the _custom_response() method
    - the human players are called via the _terminal_response() method
    - the backend players are called via the generate_response() method of the backend
    """

    def __init__(self, model: backends.Model):
        """
        Args:
            model: A backends.Model instance to be used by this Player instance.
        """
        self.model = model
        self.descriptor: str = None
        logger.info("Player %s", self.get_description())

    def get_description(self) -> str:
        """Get a description string for this Player instance.
        Returns:
            A string describing this Player instance's class name and used model.
        """
        return f"{self.__class__.__name__}, {self.model}"

    def __call__(self, messages: List[Dict], turn_idx) -> Tuple[Any, Any, str]:
        """Get a response from this Player instance's model.
        Passes a messages list and turn index to the model, creates a response dict for record logging, including
        timestamps and call duration, and returns a Player response tuple.
        Args:
            messages: A list of message dicts, containing the current conversation history to prompt the model with.
            turn_idx: The current turn index.
        Returns:
            A Player response tuple consisting of: The prompt as converted by the model backend; the full response dict
            to be used for recording/logging; the response text produced by the model, as post-processed by the model
            backend.
        """
        call_start = datetime.now()
        prompt = messages
        response = dict()
        if isinstance(self.model, backends.CustomResponseModel):
            response_text = self._custom_response(messages, turn_idx)
        elif isinstance(self.model, backends.HumanModel):
            response_text = self._terminal_response(messages, turn_idx)
        else:
            prompt, response, response_text = self.model.generate_response(messages)
        call_duration = datetime.now() - call_start
        response["clem_player"] = {
            "call_start": str(call_start),
            "call_duration": str(call_duration),
            "response": response_text,
            "model_name": self.model.get_name()
        }
        return prompt, response, response_text

    def _terminal_response(self, messages, turn_idx) -> str:
        """Response for human interaction via terminal.
        Overwrite this method to customize human inputs (model_name: human, terminal).
        Args:
            messages: A list of dicts that contain the history of the conversation.
            turn_idx: The index of the current turn.
        Returns:
            The human response as text.
        """
        latest_response = "Nothing has been said yet."
        if messages:
            latest_response = messages[-1]["content"]
        print(f"\n{latest_response}")
        user_input = input(f"Your response as {self.__class__.__name__} (turn: {turn_idx}):\n")
        return user_input

    def _custom_response(self, messages, turn_idx) -> str:
        """Response for programmatic Player interaction.
        Overwrite this method to implement programmatic behavior (model_name: mock, dry_run, programmatic, custom).
        Args:
            messages: A list of dicts that contain the history of the conversation.
            turn_idx: The index of the current turn.
        Returns:
            The programmatic response as text.
        """
        raise NotImplementedError()


class GameResourceLocator(abc.ABC):
    """
    Provides access to game specific resources and results (based on game path and results directory)

    Note: You should access resource only via the game resource locator! The locator knows how to refer to them.
    For example use: `gm.load_json("my_file")` which is located directly at your game directory `game/my_file.json`.
    You can access subdirectories by giving `gm.load_json("sub/my_file")` in `game/sub/my_file.json`.

    Makes a distinction between game files (which live in the game path specified in `self.game_path`)
    and the results files, which live in the results directory (`clembench/results` if not set otherwise)
    under `results/dialogue_pair/self.game_name/`
    """

    def __init__(self, name: str = None, path: str = None):
        """

        Args:
            game_name: name of the game (optional, because not needed for GameInstanceGenerator)
            ga,e_path: path to the game (optional, because not needed for GameScorer)
        """
        self.game_name = name  # for building results structure
        self.game_path = path  # for accessing game resources
        self.logger = logging.getLogger(self.__class__.__module__)

    # def file_path(self, file_name: str) -> str:
    #     """
    #     TODO: seems to be never used, check if removing breaks anything
    #     The absolute path to a game file. Sometimes we only need the path to a file, but not to load it.
    #     :param file_name: can be a sub-path
    #     :return: the absolute path to the file in the game directory
    #     """
    #     return file_utils.file_path(file_name, self.game_name)

    def load_instances(self, instances_name):
        """Construct instances path and return json object of the instance file.
        Args:
            game_path: Path to the game directory.
            instances_name: Name of the instances JSON file.
        Returns:
            A dict containing the contents of the given instances file.
        """
        if instances_name is None:
            instances_name = "instances"
        return file_utils.load_json(f"in/{instances_name}", self.game_path)

    def load_template(self, file_name: str) -> str:
        """Load a .template file from the game directory.
        Args:
            file_name: The name of the template file. Can have subdirectories e.g. "sub/my_file".
        Returns:
            The template file content as string.
        """
        return file_utils.load_file(file_name, self.game_path, file_ending=".template")

    def load_json(self, file_name: str) -> Dict:
        """Load a .json file from your game directory.
        Args:
            file_name: The name of the JSON file. Can have subdirectories e.g. "sub/my_file".
        Returns:
            The JSON file content as dict.
        """
        return file_utils.load_json(file_name, self.game_path)

    def load_results_json(self, file_name: str, results_dir: str, dialogue_pair: str) -> Dict:
        """Load a .json file from your game results directory.
        Args:
            file_name: The name of the JSON file. Can have subdirectories e.g. "sub/my_file".
            results_dir: The string path to the results directory.
            dialogue_pair: The name of the model pair directory. The directory name is retrieved from the results
                directory file structure by classes/methods that use this method.
        Returns:
            The JSON file content as dict.
        """
        return file_utils.load_results_json(file_name, results_dir, dialogue_pair, self.game_name)

    def load_csv(self, file_name: str) -> Dict:
        """Load a .csv file from your game directory.
        Args:
            file_name: The name of the CSV file. Can have subdirectories e.g. "sub/my_file".
        Returns:
            The CSV file content as dict.
        """
        return file_utils.load_csv(file_name, self.game_path)

    def load_file(self, file_name: str, file_ending: str = None) -> str:
        """Load an arbitrary file from your game directory.
        Args:
            file_name: The name of the file. Can have subdirectories e.g. "sub/my_file".
            file_ending: The file type suffix of the file. Optional: Can be part of file_name.
        Returns:
            The file content as string.
        """
        return file_utils.load_file(file_name, self.game_path, file_ending=file_ending)

    def store_file(self, data, file_name: str, sub_dir: str = None):
        """Store a file in your game directory.
        Args:
            data: The data to store in the file.
            file_name: The name of the file. Can have subdirectories e.g. "sub/my_file".
            sub_dir: The subdirectory to store the file in. Automatically created when given; otherwise an error will
                be thrown.
        """
        fp = file_utils.store_file(data, file_name, self.game_path, sub_dir=sub_dir)
        self.logger.info("Game file stored to %s", fp)

    def store_results_file(self, data, file_name: str, dialogue_pair: str, sub_dir: str = None, root_dir: str = None):
        """Store a results file in your game results' directory. The top-level directory is 'results'.
        Args:
            data: The data to store in the file.
            file_name: The name of the file. Can have subdirectories e.g. "sub/my_file".
            dialogue_pair: The name of the model pair directory. The directory name is retrieved from the results
                directory file structure by classes/methods that use this method.
            sub_dir: The subdirectory to store the results file in. Automatically created when given; otherwise an
                error will be thrown.
            root_dir: An (alternative) results directory structure given as a relative or absolute path.
        """
        game_results_path = file_utils.game_results_dir(root_dir, dialogue_pair, self.game_name)
        fp = file_utils.store_file(data, file_name, game_results_path, sub_dir)

        self.logger.info(f"Results file stored to {fp}")


class GameRecorder(GameResourceLocator):

    def __init__(self, game_name: str, game_path: str):
        super().__init__(game_name, game_path)
        self.log_current_turn = -1
        """ Stores players and turn during the runs """
        self.interactions = {
            "players": {},
            "turns": []
        }
        """ Stores calls to the API """
        self.requests = []

    def log_next_turn(self):
        """Call this method to group interactions per turn."""
        self.log_current_turn += 1
        self.interactions["turns"].append([])

    def log_key(self, key: str, value: Any):
        """Add a key and value to the internal log.
        Args:
            key: A string to identify the kind of log entry to be made.
            value: The content of the entry to be logged.
        """
        self.interactions[key] = value
        self.logger.info(f"{self.game_name}: Logged a game-specific interaction key: {key}.")

    def log_players(self, players_dic: Dict):
        """Log/record the players in this game episode.
        Args:
            players_dic: Dictionary of players in this game episode.
        """
        self.interactions["players"] = players_dic
        self.logger.info(f"{self.game_name}: Logged players metadata.")

    def log_event(self, from_: str, to: str, action: Dict, call: Tuple[Any, Any] = None):
        """Add an event to the internal log.
        It can be only an action or an action plus an API call that should have the same timestamp as the action.
        Args:
            from_: The identifier string of the Player/GM that originated the action.
            to: The identifier string of the Player/GM target of the action.
            action: The benchmark action to be logged.
            call: If given, this is a tuple whose first element is the input prompt object (after API-specific
                manipulation) as passed to the API and the second element is the raw response object as returned by the
                API.
        """
        assert self.log_current_turn >= 0, f"Call log_add_new_turn at least once " \
                                           f"(log_current_turn={self.log_current_turn})"
        timestamp = datetime.now().isoformat()
        action_obj = {
            "from": from_,
            "to": to,
            "timestamp": timestamp,
            "action": action
        }
        self.interactions["turns"][self.log_current_turn].append(copy.deepcopy(action_obj))
        self.logger.info(
            f"{self.game_name}: Logged {action['type']} action ({from_}->{to}).")
        if call:
            call_obj = {
                "timestamp": timestamp,
                "manipulated_prompt_obj": self._needs_copy(call[0]),
                "raw_response_obj": self._needs_copy(call[1])
            }
            self.requests.append(call_obj)
            self.logger.info(f"{self.game_name}: Logged a call with timestamp {timestamp}")

    @staticmethod
    def _needs_copy(call_obj):
        """Deepcopy objects that may otherwise lead to reference issues.
        Args:
            call_obj: The object to be deep-copied for safety.
        Returns:
            The deep-copy of the passed object, or the original object if it is safe to use.
        """
        if isinstance(call_obj, Dict) or isinstance(call_obj, List):
            return copy.deepcopy(call_obj)
        elif isinstance(call_obj, str):
            return call_obj[:]
        return call_obj

    def store_records(self, results_root: str, dialogue_pair_desc: str, game_record_dir: str):
        """Store benchmark records.
        Raise warnings if a mandatory element is empty or format is wrong.
        Args:
            results_root: The root path to the results directory.
            dialogue_pair_desc: A string combining the Player pair names to be used as directory name.
            game_record_dir: The game's record directory path.
        """
        if not self.interactions["players"]:
            self.logger.warning(f"Players metadada is missing!")
        else:
            for name in self.interactions["players"]:
                """The transcript builder relies on specific player identifiers."""
                try:
                    assert name == "GM" or name.startswith("Player ")
                except AssertionError:
                    self.logger.warning(f"Invalid player identifiers, html builder won't work.")
        if not self.interactions["turns"]:
            self.logger.warning(f"Interaction logs are missing!")
        if not self.requests:
            self.logger.warning(f"No calls logged!")
        self.store_results_file(self.interactions, "interactions.json",
                                dialogue_pair_desc,
                                sub_dir=game_record_dir,
                                root_dir=results_root)
        self.store_results_file(self.requests, "requests.json",
                                dialogue_pair_desc,
                                sub_dir=game_record_dir,
                                root_dir=results_root)


class GameMaster(GameRecorder):
    """Base class to contain game-specific functionality.

    A GameMaster (sub-)class

    - prepares a concrete game instance
    - plays an episode of a game instance
    - records a game episode
    - evaluates the game episode records
    - builds the interaction transcripts
    """

    def __init__(self, name: str, path: str, experiment: Dict, player_models: List[backends.Model] = None):
        """
        Args:
            name: The name of the game (as specified in game_registry).
            path: Path to the game (as specified in game_registry).
            experiment: The experiment (set of instances) to use.
            player_models: Player models to use for one or two players.
        """
        super().__init__(name, path)
        self.experiment: Dict = experiment
        self.player_models: List[backends.Model] = player_models

    def setup(self, **kwargs):
        """Load resources and prepare everything to play the game.
        Needs to log the players dictionary via self.log_players(players_dict).
        Called by the game's GameBenchmark run method for each game instance.
        Args:
            kwargs: Keyword arguments used to set up the GameMaster instance.
        """
        raise NotImplementedError()

    def play(self) -> None:
        """Play the game (multiple turns of a specific game instance)."""
        raise NotImplementedError()


class GameScorer(GameResourceLocator):
    """Calculates scores based on interaction logs."""
    def __init__(self, name: str, experiment: Dict, game_instance: Dict):
        """
        Args:
            name: The name of the game.
            experiment: The experiment to score.
            game_instance: The game instance to score.
        """
        super().__init__(name=name)
        self.experiment = experiment
        self.game_instance = game_instance
        """ Stores values of score computation """
        self.scores = {
            "turn scores": {},
            "episode scores": {},
        }

    def store_scores(self, results_root: str, dialogue_pair: str, game_record_dir: str):
        """Store calculated scores to scores.json file.
        Args:
            results_root: The root path to the results directory.
            dialogue_pair: A string path to the Player pair results directory.
            game_record_dir: The game's record directory path.
        """
        self.store_results_file(self.scores, "scores.json",
                                dialogue_pair=dialogue_pair,
                                sub_dir=game_record_dir,
                                root_dir=results_root)

    def log_turn_score(self, turn_idx, score_name, score_value):
        """Record a turn-level score for a single turn.
        Args:
            turn_idx: The turn index for the turn the score is to be recorded for.
            score_name: The name of the turn-level score to record.
            score_value: The value to be recorded for the turn-level score for this turn.
        """
        if isinstance(score_value, bool):
            self.logger.warning(f"{self.game_name}: Score {score_name} value is boolean, this can break the eval!")
        if turn_idx not in self.scores["turn scores"]:
            self.scores["turn scores"][turn_idx] = {}
        if score_name in self.scores["turn scores"][turn_idx]:
            self.logger.warning(f"{self.game_name}: Score {score_name} overwritten at turn {turn_idx}!")
        self.scores["turn scores"][turn_idx][score_name] = score_value
        self.logger.info(f"{self.game_name}: Logged turn {turn_idx} score {score_name}={score_value}.")

    def log_episode_score(self, score_name, score_value):
        """Record an episode-level score for a single turn.
        Args:
            score_name: The name of the episode-level score to record.
            score_value: The value to be recorded for the episode-level score.
        """
        if score_name in self.scores["episode scores"]:
            self.logger.warning(f"{self.game_name}: Episode score {score_name} overwritten!")
        self.scores["episode scores"][score_name] = score_value
        self.logger.info(f"{self.game_name}: Logged episode score {score_name}={score_value}.")

    def compute_scores(self, episode_interactions: Dict) -> None:
        """Compute and log scores for a game episode.
        This method is used to perform complete scoring of an episode.
        Args:
            episode_interactions: Dict containing the episode's interactions. This contains the actions recorded during
                a benchmark run.
        """
        self.score_turns(episode_interactions)
        self.score_game(episode_interactions)

    def score_turns(self, episode_interactions: Dict) -> None:
        """Iterate over episode turns, calculate and log turn-level scores.
        This method is intended to contain any game-specific turn-level scoring. Must be implemented!
        Use the log_turn_score method to log turn-level scores.
        Args:
            episode_interactions: Dict containing the episode's interactions. This contains the actions recorded during
                a benchmark run.
        """
        # Loop over turns, calculate and log turn-specific scores
        raise NotImplementedError()

    def score_game(self, episode_interactions: Dict) -> None:
        """Calculate and record standard clembench metric scores for an episode.
        Args:
            episode_interactions: Dict containing the episode's interactions. This contains the actions recorded during
                a benchmark run.
        """
        self.score_game_end(episode_interactions)
        self.score_requests(episode_interactions)
        self.log_main_score(episode_interactions)

    def score_game_end(self, episode_interactions: Dict) -> None:
        """Calculate and record the ABORTED, LOSE and SUCCESS standard clembench metric scores.
        Convenience method to cover mandatory clembench metric scores, so their calculation does not need to be
        implemented anew for each new clemgame.
        Args:
            episode_interactions: Dict containing the episode's interactions. This contains the actions recorded during
                a benchmark run.
        """
        aborted = int(episode_interactions[ms.METRIC_ABORTED])
        lose = int(episode_interactions[ms.METRIC_LOSE]) if not aborted else 0
        success = 1 - lose if not aborted else 0

        self.log_episode_score(ms.METRIC_ABORTED, aborted)
        self.log_episode_score(ms.METRIC_LOSE, lose)
        self.log_episode_score(ms.METRIC_SUCCESS, success)

    def score_requests(self, episode_interactions: Dict):
        """Calculate and record standard request-based clembench metric scores.
        Records total request count, parsed, violated, and success ratio of parsed requests over all requests in an
        episode.
        Convenience method to cover mandatory clembench metric scores, so their calculation does not need to be
        implemented anew for each new clemgame.
        Args:
            episode_interactions: Dict containing the episode's interactions. This contains the actions recorded during
                a benchmark run.
        """
        request_count = episode_interactions[
            ms.METRIC_REQUEST_COUNT]  # could also be calculated by adding parsed and violated requests
        parsed_requests = episode_interactions[ms.METRIC_REQUEST_COUNT_PARSED]
        violated_requests = episode_interactions[ms.METRIC_REQUEST_COUNT_VIOLATED]

        self.log_episode_score(ms.METRIC_REQUEST_COUNT, request_count)
        self.log_episode_score(ms.METRIC_REQUEST_COUNT_PARSED, parsed_requests)
        self.log_episode_score(ms.METRIC_REQUEST_COUNT_VIOLATED, violated_requests)
        self.log_episode_score(ms.METRIC_REQUEST_SUCCESS, parsed_requests / request_count)

    def log_main_score(self, episode_interactions: Dict):
        """Record the game's main score.
        Replace this method with a method that calculates and logs the clemgame's main score aka BENCH_SCORE.
        Must be implemented! Recording BENCH_SCORE is mandatory.
        Args:
            episode_interactions: Dict containing the episode's interactions. This contains the actions recorded during
                a benchmark run.
        """
        raise NotImplementedError()


class DialogueGameMaster(GameMaster):
    """Extended GameMaster, implementing turns as described in the clembench paper.
    Has most logging and gameplay procedures implemented, including convenient logging methods.
    """
    def __init__(self, name: str, path: str, experiment: dict, player_models: List[backends.Model]):
        """
        Args:
            name: The name of the game (as specified in game_registry).
            path: Path to the game (as specified in game_registry).
            experiment: The experiment (set of instances) to use.
            player_models: Player models to use for one or two players.
        """
        super().__init__(name, path, experiment, player_models)
        # the logging works with an internal mapping of "Player N" -> Player
        self.players_by_names: Dict[str, Player] = collections.OrderedDict()
        self.messages_by_names: Dict[str, List] = dict()
        self.current_turn: int = 0

    def get_players(self) -> List[Player]:
        """Get a list of the players.
        Returns:
            List of Player instances in the order they are added.
        """
        return list(self.players_by_names.values())

    def add_player(self, player: Player):
        """Add a player to the game.
        Note: The players will be called in the same order as added!
        Args:
            player: The player to be added to the game.
        """
        idx = len(self.players_by_names)
        player.descriptor = f"Player {idx + 1}"
        self.players_by_names[player.descriptor] = player
        self.messages_by_names[player.descriptor] = []

    def setup(self, **kwargs):
        """Load resources and prepare everything to play the game.
        Needs to log the players dictionary via self.log_players(players_dict).
        Intended to be left as-is by inheriting classes. Implement game-specific setup functionality in the _on_setup
        method.
        Called by the game's GameBenchmark run method for each game instance.
        Args:
            kwargs: Keyword arguments used to set up the GameMaster instance. This is usually a game instance object
                read from the game's instances.json.
        """
        self._on_setup(**kwargs)
        # log players
        players_descriptions = collections.OrderedDict(GM=f"Game master for {self.game_name}")
        for name, player in self.players_by_names.items():
            players_descriptions[name] = player.get_description()
        # log player ID and description dcit:
        self.log_players(players_descriptions)

    def _on_setup(self, **kwargs):
        """Method executed at the start of the default setup method.
        Template method: Must be implemented!
        Use add_player() here to add the players.
        Args:
            kwargs: Keyword arguments of the game instance. This is usually a game instance object
                read from the game's instances.json.
        """
        raise NotImplementedError()

    def play(self) -> None:
        """Main play loop method.
        This method is called to run the game for benchmarking.
        Intended to be left as-is by inheriting classes. Implement additional game functionality in the
        _on_before_game, _does_game_proceed, _on_before_turn, _should_reprompt, _on_before_reprompt, _on_after_turn and
        _on_after_game methods.
        """
        self._on_before_game()
        inner_break = False
        while not inner_break and self._does_game_proceed():
            self.log_next_turn()  # not sure if we want to do this always here (or add to _on_before_turn)
            self._on_before_turn(self.current_turn)
            self.logger.info(f"{self.game_name}: %s turn: %d", self.game_name, self.current_turn)
            for player in self.__player_sequence():
                if not self._does_game_proceed():
                    inner_break = True  # break outer loop without calling _does_game_proceed again
                    break  # potentially stop in between player turns
                self.prompt(player)
                while self._should_reprompt(player):
                    self._on_before_reprompt(player)
                    self.prompt(player, is_reprompt=True)
            self._on_after_turn(self.current_turn)
            self.current_turn += 1
        self._on_after_game()

    def prompt(self, player: Player, is_reprompt=False):
        """Prompt a player model.
        Includes logging of 'send message' and 'get message' actions.
        Intended to be left as-is by inheriting classes. Implement game-specific functionality in the
        _should_reprompt, _on_before_reprompt, _after_add_player_response, _validate_player_response and
        _on_parse_response methods.
        Args:
            player: The Player instance to be prompted.
            is_reprompt: If this is a reprompt attempt. This is intended for re-prompting with modified prompts.
        """
        # GM -> Player
        history = self.messages_by_names[player.descriptor]
        assert history, f"messages history must not be empty for {player.descriptor}"

        last_entry = history[-1]
        assert last_entry["role"] != "assistant", "Last entry should not be assistant " \
                                                  "b.c. this would be the role of the current player"
        message = last_entry["content"]

        action_type = 'send message' if not is_reprompt else 'send message (reprompt)'
        action = {'type': action_type, 'content': message}
        self.log_event(from_='GM', to=player.descriptor, action=action)

        _prompt, _response, response_message = player(history, self.current_turn)

        # Player -> GM
        action = {'type': 'get message', 'content': response_message}
        # log 'get message' event including backend/API call:
        self.log_event(from_=player.descriptor, to="GM", action=action, call=(_prompt, _response))

        # GM -> GM
        self.__validate_parse_and_add_player_response(player, response_message)

    def _should_reprompt(self, player: Player):
        """Method to check if a Player should be re-prompted.
        This is intended to check for invalid responses.
        Args:
            player: The Player instance to re-prompt.
        """
        return False

    def _on_before_reprompt(self, player: Player):
        """Method executed before reprompt is passed to a Player.
        Hook
        Change the prompt to reprompt the player on e.g. an invalid response.
        Add the new prompt to the players message via self.add_user_message(player, new_prompt)
        Args:
            player: The Player instance that produced the invalid response.
        """
        pass

    def log_message_to(self, player: Player, message: str):
        """Logs a 'send message' action from GM to Player.
        This is a logging method, and will not add the message to the conversation history on its own!
        Args:
            player: The Player instance the message is targeted at.
            message: The message content sent to the Player instance.
        """
        action = {'type': 'send message', 'content': message}
        self.log_event("GM", player.descriptor, action)

    def log_message_to_self(self, message: str):
        """Logs a 'metadata' action from GM to GM.
        This is a logging method, and will not add anything to the conversation history.
        Args:
            message: The message content logged as metadata.
        """
        action = {'type': 'metadata', 'content': message}
        self.log_event("GM", "GM", action)

    def log_to_self(self, type_: str, value: str):
        """Logs an action of the passed type from GM to GM.
        This is a logging method, and will not add anything to the conversation history.
        Args:
            type_: The type of the action to be logged.
            value: The content value of the action to be logged.
        """
        action = {'type': type_, 'content': value}
        self.log_event("GM", "GM", action)

    def add_message(self, player: Player, utterance: str, role: str):
        """Adds a message to the conversation history.
        This method is used to iteratively create the conversation history, but will not log/record messages
        automatically.
        Args:
            player: The Player instance that produced the message. This is usually a model output, but can be the game's
                GM as well, if it directly adds messages to the conversation history. TODO: Check use
            utterance: The text content of the message to be added.
            role: The chat/instruct conversation role to use for this message. Either 'user' or 'assistant', or 'system'
                for models/templates that support it. This is important to properly apply chat templates. Some chat
                templates require that roles always alternate between messages!
        """
        message = {"role": role, "content": utterance}
        history = self.messages_by_names[player.descriptor]
        history.append(message)

    def add_user_message(self, player: Player, utterance: str):
        """Adds a message with the 'user' role to the conversation history.
        This method is to be used for 'user' messages, usually the initial prompt and GM response messages. Used to
        iteratively create the conversation history, but will not log/record messages automatically.
        Args:
            player: The Player instance that produced the message. This is usually the game's GM, if it directly adds
                messages to the conversation history. TODO: Check use
            utterance: The text content of the message to be added.
        """
        self.add_message(player, utterance, role="user")

    def add_assistant_message(self, player: Player, utterance: str):
        """Adds a message with the 'assistant' role to the conversation history.
        This method is to be used for 'assistant' messages, usually model outputs. Used to iteratively create the
        conversation history, but will not log/record messages automatically.
        Args:
            player: The Player instance that produced the message.
            utterance: The text content of the message to be added.
        """
        self.add_message(player, utterance, role="assistant")

    def __validate_parse_and_add_player_response(self, player: Player, utterance: str):
        """Checks player response validity, parses it and adds it to the conversation history.
        Part of the play loop, not intended to be modified - modify _validate_player_response, _on_parse_response and/or
        _after_add_player_response instead.
        Args:
            player: The Player instance that produced the response.
            utterance: The text content of the response.
        """
        # todo: it seems we should change the order here: Parse should come first, and then validate.
        # While parse might throw a parsing (format error) validate would check solely for satisfied game rules.
        # Note: this would allow to cut off too long responses (during parse) and to only validate on the cut off piece.
        if self._validate_player_response(player, utterance):
            utterance = self.__parse_response(player, utterance)
            self.add_assistant_message(player, utterance)
            self._after_add_player_response(player, utterance)

    def _after_add_player_response(self, player: Player, utterance: str):
        """Method executed after a player response has been validated and added to the conversation history.
        Hook: Modify this method for game-specific functionality.
        Add the utterance to other player's history, if necessary. To do this use the method
        add_user_message(other_player,utterance).
        Args:
            player: The Player instance that produced the response (or has been modified by the GM).
            utterance: The text content of the message that was added.
        """
        pass

    def _validate_player_response(self, player: Player, utterance: str) -> bool:
        """Decide if an utterance should be added to the conversation history.
        Hook: Modify this method for game-specific functionality.
        This is also the place to check for game end conditions.
        Args:
            player: The Player instance for which the response is added as "assistant" to the history.
            utterance: The text content of the message to be added.
        Returns:
            True, if the utterance is fine; False, if the response should not be added to the history.
        """
        return True

    def __parse_response(self, player: Player, utterance: str) -> str:
        """Parses a response and logs the message parsing result.
        Part of the validate-parse loop, not intended to be modified - modify _on_parse_response instead.
        Args:
            player: The Player instance that produced the response.
            utterance: The text content of the response.
        Returns:
            The response content, potentially modified by the _on_parse_response method.
        """
        _utterance, log_action = self._on_parse_response(player, utterance)
        if _utterance == utterance:
            return utterance
        if log_action:
            action = {'type': 'parse', 'content': _utterance}
            self.log_event(from_="GM", to="GM", action=action)
        return _utterance

    def _on_parse_response(self, player: Player, utterance: str) -> Tuple[str, bool]:
        """Decide if a response utterance should be modified and apply modifications.
        Hook: Modify this method for game-specific functionality.
        If no modifications are applied, this method must simply return a tuple of the utterance and True.
        When a modified utterance and a true value is returned, then a 'parse' event is logged.
        Args:
            player: The Player instance that produced the response. Intended to allow for individual handling of
                different players.
            utterance: The text content of the response.
        Returns:
            A tuple of the (modified) utterance, and a bool to determine if the parse action is to be logged (default:
            True).
        """
        return utterance, True

    def _on_before_turn(self, turn_idx: int):
        """Executed in play loop after turn advance and before proceed check and prompting.
        Hook: Modify this method for game-specific functionality.
        Args:
            turn_idx: The current turn index.
        """
        pass

    def _on_after_turn(self, turn_idx: int):
        """Executed in play loop after prompting.
        Hook: Modify this method for game-specific functionality.
        Args:
            turn_idx: The current turn index.
        """
        pass

    def __player_sequence(self) -> List[Player]:
        """Return players in the order they are added.
        Returns:
            List of Player instances in the order they are added.
        """
        return self.get_players()

    def _does_game_proceed(self) -> bool:
        """Check if game should proceed.
        Template method: Must be implemented!
        This method is used to determine if a game should continue or be stopped. Both successful completion of the game
        and game-ending failures should lead to this method returning False.
        Returns:
            A bool, True if game continues, False if game should stop.
        """
        raise NotImplementedError()

    def _on_before_game(self):
        """Executed once at the start, before entering the play loop.
        Hook: Modify this method for game-specific functionality.
        Adding the initial prompt to the dialogue history with this method is recommended.
        """
        pass

    def _on_after_game(self):
        """Executed once at the end, after exiting the play loop.
        Hook: Modify this method for game-specific functionality.
        This method is useful to process and log/record overall game results.
        """
        pass


class GameBenchmark(GameResourceLocator):
    """Organizes the run of a particular collection of game instances which compose a benchmark for the game.
    Supports different experiment conditions for games.
    """

    def __init__(self, game_spec: GameSpec):
        """
        Args:
            game_spec: The name of the game (as specified in game_registry)
        """
        super().__init__(game_spec["game_name"], game_spec["game_path"])
        self.instances = None
        self.filter_experiment: List[str] = []
        self.is_single_player = True if game_spec["players"] == "one" else False

    def setup(self, instances_name: str = None):
        """Set up a benchmark run of a clemgame.
        Args:
            game_path: Path to the game directory.
            instances_name: Name of the instances JSON file to be used for the benchmark run.
        """
        self.instances = self.load_instances(instances_name)

    def build_transcripts(self, results_dir: str):
        """Create and store readable HTML and LaTeX episode transcripts.
        Transcripts are stored in each corresponding episode directory.
        Args:
            results_dir: Path to the results directory.
        """
        results_root = file_utils.results_root(results_dir)
        dialogue_partners = [model_dir for model_dir in os.listdir(results_root)
                             if os.path.isdir(os.path.join(results_root, model_dir))]
        for dialogue_pair in dialogue_partners:
            game_result_path = file_utils.game_results_dir(results_root, dialogue_pair, self.game_name)
            if not os.path.exists(game_result_path) or not os.path.isdir(game_result_path):
                stdout_logger.info("No results directory found at: " + game_result_path)
                continue

            experiment_dirs = [experiment_dir for experiment_dir in os.listdir(game_result_path)
                               if os.path.isdir(os.path.join(game_result_path, experiment_dir))]
            if not experiment_dirs:
                stdout_logger.warning(f"{self.game_name}: No experiments for {dialogue_pair}")
            for experiment_dir in experiment_dirs:
                experiment_path = os.path.join(game_result_path, experiment_dir)
                experiment_name = "_".join(experiment_dir.split("_")[1:])  # remove leading index number
                if self.filter_experiment and experiment_name not in self.filter_experiment:
                    stdout_logger.info(f"Skip experiment {experiment_name}")
                    continue
                stdout_logger.info(f"Transcribe: {experiment_name}")
                experiment_config = self.load_results_json(f"{experiment_dir}/experiment_{experiment_name}",
                                                           results_root, dialogue_pair)
                episode_dirs = [file for file in os.listdir(experiment_path)
                                if os.path.isdir(os.path.join(experiment_path, file))]
                error_count = 0
                for episode_dir in tqdm(episode_dirs, desc="Building transcripts"):
                    try:
                        rel_episode_path = f"{experiment_dir}/{episode_dir}"
                        game_instance = self.load_results_json(f"{rel_episode_path}/instance",
                                                               results_root, dialogue_pair)
                        game_interactions = self.load_results_json(f"{rel_episode_path}/interactions",
                                                                   results_root, dialogue_pair)

                        transcript = transcript_utils.build_transcript(game_interactions, experiment_config,
                                                                       game_instance, dialogue_pair)
                        self.store_results_file(transcript, "transcript.html",
                                                dialogue_pair,
                                                sub_dir=rel_episode_path,
                                                root_dir=results_root)
                        transcript_tex = transcript_utils.build_tex(game_interactions)
                        self.store_results_file(transcript_tex, "transcript.tex",
                                                dialogue_pair,
                                                sub_dir=rel_episode_path,
                                                root_dir=results_root)
                    except Exception:  # continue with other episodes if something goes wrong
                        self.logger.exception(f"{self.game_name}: Cannot transcribe {episode_dir} (but continue)")
                        error_count += 1
                if error_count > 0:
                    stdout_logger.error(
                        f"{self.game_name}: '{error_count}' exceptions occurred: See clembench.log for details.")

    def compute_scores(self, results_dir: str):
        """Compute and store scores for each episode and player pair.
        Episode score JSON files are stored in each corresponding episode directory. Combined scores for a player/model
        pair are stored in the player pair directory.
        Args:
            results_dir: Path to the results directory.
        """
        results_root = file_utils.results_root(results_dir)
        dialogue_partners = [model_dir for model_dir in os.listdir(results_root)
                             if os.path.isdir(os.path.join(results_root, model_dir))]
        for dialogue_pair in dialogue_partners:
            game_result_path = file_utils.game_results_dir(results_root, dialogue_pair, self.game_name)
            if not os.path.exists(game_result_path) or not os.path.isdir(game_result_path):
                stdout_logger.info("No results directory found at: " + game_result_path)
                continue

            experiment_dirs = [experiment_dir for experiment_dir in os.listdir(game_result_path)
                               if os.path.isdir(os.path.join(game_result_path, experiment_dir))]
            if not experiment_dirs:
                stdout_logger.warning(f"{self.game_name}: No experiments for {dialogue_pair}")
            for experiment_dir in experiment_dirs:
                experiment_path = os.path.join(game_result_path, experiment_dir)
                experiment_name = "_".join(experiment_dir.split("_")[1:])  # remove leading index number
                if self.filter_experiment and experiment_name not in self.filter_experiment:
                    stdout_logger.info(f"Skip experiment {experiment_name}")
                    continue
                stdout_logger.info(f"Scoring: {experiment_name}")
                experiment_config = self.load_results_json(f"{experiment_dir}/experiment_{experiment_name}",
                                                           results_root, dialogue_pair)
                episode_dirs = [file for file in os.listdir(experiment_path)
                                if os.path.isdir(os.path.join(experiment_path, file))]
                error_count = 0
                for episode_dir in tqdm(episode_dirs, desc="Scoring episodes"):
                    try:
                        rel_episode_path = f"{experiment_dir}/{episode_dir}"
                        game_instance = self.load_results_json(f"{rel_episode_path}/instance",
                                                               results_root, dialogue_pair)
                        game_interactions = self.load_results_json(f"{rel_episode_path}/interactions",
                                                                   results_root, dialogue_pair)

                        game_scorer = self.create_game_scorer(experiment_config, game_instance)
                        game_scorer.compute_scores(game_interactions)
                        game_scorer.store_scores(results_root, dialogue_pair, rel_episode_path)
                    except Exception:  # continue with other episodes if something goes wrong
                        self.logger.exception(f"{self.game_name}: Cannot score {episode_dir} (but continue)")
                        error_count += 1
                if error_count > 0:
                    stdout_logger.error(
                        f"{self.game_name}: '{error_count}' exceptions occurred: See clembench.log for details.")

    def run(self, player_models: List[backends.Model], results_dir: str):
        """Runs game-play on all game instances for a game.
        There must be an instances.json with the following structure:
        "experiments": [ # this is required
            {
                "name": <experiment-name>, # this is required
                "param1": "value1", # optional
                "param2": "value2", # optional
                "game_instances": [ # this is required
                    {"game_id": <value>, "initial_prompt": ... },
                    {"game_id": <value>, "initial_prompt": ... }
                ]
            }
        ]

        The instances will be automatically stored in "game-name" with the following structure:
            - results
                - pairing
                    - game-name
                        - experiment_name
                            - experiment.json
                            - episode_id
                                - instance.json
                                - interaction.json
                                
        Args:
            player_models: A list of backends.Model instances to run the game with.
            results_dir: Path to the results directory.
        """
        results_root = file_utils.results_root(results_dir)
        experiments: List = self.instances["experiments"]
        if not experiments:
            self.logger.warning(f"{self.game_name}: No experiments for %s", self.game_name)
        total_experiments = len(experiments)
        for experiment_idx, experiment in enumerate(experiments):
            experiment_name = experiment['name']
            if self.filter_experiment and experiment_name not in self.filter_experiment:
                stdout_logger.info(f"Skip experiment {experiment_idx + 1} of {total_experiments}: {experiment_name}")
                continue
            stdout_logger.info(f"Run experiment {experiment_idx + 1} of {total_experiments}: {experiment_name}")
            # Determine dialogue partners: How often to run the experiment with different partners
            dialogue_partners: List[List[backends.Model]] = []

            if player_models:  # favor runtime argument over experiment config
                dialogue_partners = [player_models]
            elif "dialogue_partners" in experiment:  # edge-case when names are given in experiment config
                for dialogue_pair_names in experiment["dialogue_partners"]:
                    player_models = []
                    for model_name in dialogue_pair_names:
                        player_model = backends.get_model_for(model_name)
                        player_models.append(player_model)
                    dialogue_partners.append(player_models)
                self.logger.info(f"{self.game_name}: Detected 'dialogue_partners' in experiment config. "
                                 f"Will run with: {dialogue_partners}")

            if not dialogue_partners:
                message = (f"{self.game_name}: Neither 'dialogue_partners' set in experiment instance"
                           f" nor 'models' given as run arg")
                stdout_logger.error(message)
                raise ValueError(message)

            for dialogue_pair in dialogue_partners:
                if self.is_single_player:
                    if len(dialogue_pair) > 1:
                        message = f"Too many player for singe-player game '{self.game_name}': '{len(dialogue_partners)}'"
                        stdout_logger.error(message)
                        raise ValueError(message)
                    model_0 = dialogue_pair[0]
                    model_0 = f"{model_0.get_name()}-t{model_0.get_temperature()}"
                    # still we store to model--model dir (virtual self-play)
                    dialogue_pair_desc = f"{model_0}--{model_0}"
                else:  # 2-players
                    if len(dialogue_pair) > 2:
                        message = f"Too many player for two-player game '{self.game_name}': '{len(dialogue_partners)}'"
                        stdout_logger.error(message)
                        raise ValueError(message)
                    if len(dialogue_pair) == 1:
                        dialogue_pair.append(dialogue_pair[0])  # model expansion
                    model_0 = dialogue_pair[0]
                    model_0 = f"{model_0.get_name()}-t{model_0.get_temperature()}"
                    model_1 = dialogue_pair[1]
                    model_1 = f"{model_1.get_name()}-t{model_1.get_temperature()}"
                    dialogue_pair_desc = f"{model_0}--{model_1}"
                episode_counter = 0

                self.logger.info("Activity: %s Experiment: %s Partners: %s Episode: %d",
                                 self.game_name, experiment_name, dialogue_pair_desc, episode_counter)

                experiment_record_dir = f"{experiment_idx}_{experiment_name}"
                experiment_config = {k: experiment[k] for k in experiment if k != 'game_instances'}

                # Add some important infos to track
                experiment_config["timestamp"] = datetime.now().isoformat()
                experiment_config["dialogue_partners"] = dialogue_pair_desc

                self.store_results_file(experiment_config,
                                        f"experiment_{experiment_name}.json",
                                        dialogue_pair_desc,
                                        sub_dir=experiment_record_dir,
                                        root_dir=results_root)

                error_count = 0
                time_experiment_start = datetime.now()
                game_instances: List = experiment["game_instances"]
                for game_instance in tqdm(game_instances, desc="Playing games"):
                    game_id = game_instance["game_id"]
                    self.logger.info("Activity: %s Experiment: %s Episode: %d Game: %s",
                                     self.game_name, experiment_name, episode_counter, game_id)
                    episode_dir = experiment_record_dir + f"/episode_{episode_counter}"
                    self.store_results_file(game_instance,
                                            f"instance.json",
                                            dialogue_pair_desc,
                                            sub_dir=episode_dir,
                                            root_dir=results_root)
                    try:
                        game_master = self.create_game_master(experiment_config, dialogue_pair)
                        game_master.setup(**game_instance)
                        game_master.play()
                        game_master.store_records(results_root, dialogue_pair_desc, episode_dir)
                    except Exception:  # continue with other episodes if something goes wrong
                        self.logger.exception(f"{self.game_name}: Exception for episode {game_id} (but continue)")
                        error_count += 1
                    episode_counter += 1
                if error_count > 0:
                    stdout_logger.error(
                        f"{self.game_name}: '{error_count}' exceptions occurred: See clembench.log for details.")
                # Add experiment duration and overwrite file
                time_experiment_end = datetime.now() - time_experiment_start
                experiment_config["duration"] = str(time_experiment_end)
                self.store_results_file(experiment_config,
                                        f"experiment_{experiment_name}.json",
                                        dialogue_pair_desc,
                                        sub_dir=experiment_record_dir,
                                        root_dir=results_root)

    def create_game_master(self, experiment: Dict, player_models: List[backends.Model]) -> GameMaster:
        """Create a game-specific GameMaster subclass instance to run the game with.
        Must be implemented!
        Args:
            experiment: The experiment (set of instances) to run.
            player_models: Player models to use for one or two players.
        Returns:
            A game-specific GameMaster subclass instance.
        """
        raise NotImplementedError()

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        """Create a game-specific GameScorer subclass instance to score benchmark records with.
        Must be implemented!
        Args:
            experiment: The experiment (set of instances) to score.
            game_instance: The game instance to score.
        Returns:
            A game-specific GameScorer subclass instance.
        """
        raise NotImplementedError()


class GameInstanceGenerator(GameResourceLocator):
    """Create all game instances for a game benchmark.
    Results in an instances.json with the following structure:

    "experiments": [ # this is required
        {
            "name": <experiment-name>, # this is required
            "param1": "value1", # optional
            "param2": "value2", # optional
            "game_instances": [ # this is required
                {"id": <value>, "initial_prompt": ... },
                {"id": <value>, "initial_prompt": ... }
            ]
        }
    ]
    """

    def __init__(self, path: str):
        """
        Args:
            path: The path to the game.
        """
        super().__init__(path=path)
        self.instances = dict(experiments=list())

    def add_experiment(self, experiment_name: str, dialogue_partners: List[Tuple[str, str]] = None) -> Dict:
        """Add an experiment to the game benchmark.
        Experiments are sets of instances, usually with different experimental variables than other experiments in a
        game benchmark.
        Call this method and adjust the returned dict to configure the experiment.
        For game instances use add_game_instance!
        Args:
            experiment_name: Name of the new game experiment.
            dialogue_partners: A list of partner definitions for which the experiment will run.
        Returns:
            A new game experiment dict.
        """
        experiment = collections.OrderedDict(name=experiment_name)
        if dialogue_partners:
            experiment["dialogue_partners"] = dialogue_partners
        experiment["game_instances"] = list()
        self.instances["experiments"].append(experiment)
        return experiment

    def add_game_instance(self, experiment: Dict, game_id):
        """Add an instance to an experiment.
        An instance holds all data to run a single episode of a game.
        Call this method and adjust the returned dict to configure the instance.
        Args:
            experiment: The experiment to which a new game instance should be added.
            game_id: Identifier of the new game instance.
        Returns:
            A new game instance dict.
        """
        game_instance = dict(game_id=game_id)
        experiment["game_instances"].append(game_instance)
        return game_instance

    def on_generate(self, **kwargs):
        """Game-specific instance generation.
        This method is intended for creation of instances and experiments for a game benchmark. Use the add_experiment
        and add_game_instance methods to create the game benchmark.
        Must be implemented!
        Args:
            kwargs: Keyword arguments (or dict) with data controlling instance generation.
        """
        raise NotImplementedError()

    def generate(self, filename="instances.json", **kwargs):
        """Generate the game benchmark and store the instances JSON file.
        Intended to not be modified by inheriting classes, modify on_generate instead.
        Args:
            filename: The name of the instances JSON file to be stored in the 'in' subdirectory. Defaults to
                'instances.json'.
            kwargs: Keyword arguments (or dict) to pass to the on_generate method.
        """
        self.on_generate(**kwargs)
        self.store_file(self.instances, filename, sub_dir="in")


def is_game(obj):
    """Check whether a class inherited from GameBenchmark.
    Args:
        obj: The object instance to check.
    Returns:
        True if the passed object is a subclass of GameBenchmark, False otherwise.
    """
    if inspect.isclass(obj) and issubclass(obj, GameBenchmark) and obj is not GameBenchmark:
        return True
    return False


def load_game(game_spec: GameSpec, do_setup: bool = True, instances_name: str = None) -> GameBenchmark:
    """Load a clemgame using a GameSpec.
    Args:
        game_spec: A GameSpec instance holding specific clemgame data.
        do_setup: Determines if the clemgame's setup method will be executed upon loading.
        instances_name: The name of the instances file to be used for the clemgame's setup if do_setup is True.
    """
    # append game directory to system path for loading game specific dependencies
    sys.path.insert(0, game_spec.game_path)
    # load game module from this master file
    spec = importlib.util.spec_from_file_location(game_spec["game_name"], game_spec.get_game_file())
    game_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(game_module)

    # extract game class from master.py (is_game checks inheritance from GameBenchmark)
    game_subclasses = inspect.getmembers(game_module, predicate=is_game)
    if len(game_subclasses) == 0:
        raise LookupError(f"There is no GameBenchmark defined in {game_module}. "
                          f"Create such a class and try again.")
    if len(game_subclasses) > 1:
        raise LookupError(f"There is more than one Game defined in {game_module}.")
    game_class_name, game_class = game_subclasses[0]
    game_cls = game_class(game_spec)  # instantiate the specific game class

    if do_setup:
        game_cls.setup(instances_name)

    return game_cls
