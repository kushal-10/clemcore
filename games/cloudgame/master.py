# TODO add to _validate_player_response: do not automatically return True (important for when not mock)
# TODO add played or aborted metric to compute_scores (see prev. todo)

from typing import List, Dict

import numpy as np

import clemgame.metrics as ms
from clemgame.clemgame import GameMaster, GameBenchmark, DialogueGameMaster, GameScorer
from clemgame import get_logger
from clemgame.clemgame import Player
from clemgame.metrics import METRIC_ABORTED, METRIC_SUCCESS, METRIC_LOSE, BENCH_SCORE, METRIC_REQUEST_COUNT, METRIC_REQUEST_COUNT_PARSED,  METRIC_REQUEST_COUNT_VIOLATED, METRIC_REQUEST_SUCCESS
from games.cloudgame.instancegenerator import GAME_NAME
from backends import Model 

logger = get_logger(__name__)

# Player1 - Multimodal Model
class Speaker(Player):
    def __init__(self, model: Model):
        super().__init__(model)

    def _custom_response(self) -> str:
        return "Maybe"
    
# Player2 - Programmatic
class Judge(Player):

    def __init__(self, model: Model):
        super().__init__(model)

    def _custom_response(self):
        return "That seems right."

class Cloudgame(DialogueGameMaster):
    """Implement mechanisms for playing Cloudgame."""

    def __init__(self, experiment: Dict, player_models: List[Model]):
        super().__init__(GAME_NAME, experiment, player_models)
        # fetch experiment parameters here
        self.max_words = 2
        self.turns = []
        self.allowed_words = ["yes", "no"]
        self.success = True
        self.aborted: bool = False

        self.experiment = experiment['name']
        self.player_backends = player_models
       

    def _on_setup(self, **game_instance):
        '''
        Sets the information specified for each instance, along with some general prompts
        '''

        self.game_instance = game_instance
        self.image = game_instance["image"]
        self.prompt_text = game_instance["prompt"]
        self.general_prompt = "This seems correct."
        self.prompt_judge = "Do you think this is correct?"
        self.second_prompt = "Are there any chickens in the image? Answer with only \"Yes\" or \"No\"."

        self.speaker = Speaker(self.player_backends[0])
        self.judge = Judge(self.player_backends[0]) # Argument hier ist relativ arbiträr
        # Change from self.experiment to self.player_backends[0] // from type str to type Model

        self.add_player(self.speaker)
        self.add_player(self.judge)


    def _does_game_proceed(self):
        if not self.aborted and len(self.turns) <= 1: # Stop at 2nd Turn (Cloud question + Chickens question)
            return True
        return False
    

    def _on_before_turn(self, turn_idx: int):
        if turn_idx == 0: # Add cloud question to speaker and programmatic question to judge
            self.add_user_message(self.speaker, self.prompt_text, image=self.image)
            self.add_user_message(self.judge, self.prompt_judge, image=self.image)
        if turn_idx == 1: # Add chicken question to speaker and programmatic question to judge
            self.add_user_message(self.speaker, self.second_prompt, image=self.image)
            self.add_user_message(self.judge, self.prompt_judge, image=self.image)

    def _validate_player_response(self, player: Player, answer: str) -> bool:
        """Check if the utterance conforms to rules (cloudgame specific)."""

        # there should never be a chicken in a picture
        if len(self.turns) != 0:
            true_answer = "no"
        
        if player == self.speaker:
            true_answer = self.experiment
            split_answer = answer.split() # Handle blank space

            # Only one word answer allowed
            if len(split_answer) != 1:
                self.success = False
                self.aborted = True
                self.log_to_self("Invalid word count", "Game aborted.")
                return False
    
            # Answer is one word but not yes/no
            if split_answer[0].lower() not in self.allowed_words:
                self.success = False
                self.aborted = True
                self.log_to_self("Invalid words", "Game aborted.")
                return False

            # Is answer correct?
            elif split_answer[0].lower() != true_answer:
                self.success = False
            
            # Correct Answer
            self.log_to_self("Valid format", "Continue")

        return True

    
    def _after_add_player_response(self, player: Player, utterance: str):
        # Add a general message
        # if player == self.judge:
        self.add_user_message(player, self.general_prompt)
        
    def _on_after_turn(self, turn_idx: int):
        self.log_to_self(type_ = "judgement", value = self.success)
        if self.aborted:
            self.log_to_self(type_ = "aborted", value = self.aborted)
        self.turns.append(self.success)


    '''
    Multimodal specific functions 
    Add image as an argument.
    '''

    def add_message(self, player: Player, utterance: str, role: str, image = None):
        if image is None: # For adding general_prompt to the speaker
            message = {"role": role, "content": utterance}
        else:
            message = {"role": role, "content": utterance, "image": image}
        history = self.messages_by_names[player.descriptor]
        history.append(message)

    def add_user_message(self, player: Player, utterance: str, image = None):
        self.add_message(player, utterance, role="user", image= image)
        

class CloudgameScorer(GameScorer):
    def __init__(self, experiment: Dict, game_instance: Dict):
        super().__init__(GAME_NAME, experiment, game_instance)      

    def compute_scores(self, episode_interactions: Dict) -> None:

        all_turn_scores = []
        for turn_idx, turn in enumerate(episode_interactions["turns"]):
            # player_1_message = turn[1]['action']['content']
            score = 0
            turn_score_dict = {"request_count": 0, "violated_request_count": 0, "parsed_request_count": 0}
            aborted = False

            for event in turn:
                action = event["action"]

                if action["type"] == "get message":
                    turn_score_dict["request_count"] += 1
                if action["type"] == "Valid format":
                    turn_score_dict["parsed_request_count"] += 1
                if action["type"] == "Invalid word count":
                    turn_score_dict["violated_request_count"] += 1
                    aborted = True
                if action["type"] == "Invalid words":
                    turn_score_dict["violated_request_count"] = 1
                    aborted = True
                if action["type"] == "judgement":
                    score = action["content"]

         # log turn request scores   
            self.log_turn_score(turn_idx, METRIC_REQUEST_COUNT_VIOLATED, turn_score_dict["violated_request_count"])
            self.log_turn_score(turn_idx, METRIC_REQUEST_COUNT_PARSED, turn_score_dict["parsed_request_count"])
            self.log_turn_score(turn_idx, METRIC_REQUEST_COUNT, turn_score_dict["request_count"])

            all_turn_scores.append(turn_score_dict)

        violated_request_count = sum([turn["violated_request_count"] for turn in all_turn_scores])
        self.log_episode_score(ms.METRIC_REQUEST_COUNT_VIOLATED, violated_request_count)
        parsed_request_count = sum([turn["parsed_request_count"] for turn in all_turn_scores])
        self.log_episode_score(ms.METRIC_REQUEST_COUNT_PARSED, parsed_request_count)
        request_count = sum([turn["request_count"] for turn in all_turn_scores])
        self.log_episode_score(ms.METRIC_REQUEST_COUNT, request_count)

        if aborted:
            self.log_episode_score(METRIC_ABORTED, 1)
            self.log_episode_score(METRIC_SUCCESS, 0)
            self.log_episode_score(METRIC_LOSE, 0)
            # Game-specific metrics
            self.log_episode_score(BENCH_SCORE, np.nan)
        else:
            self.log_episode_score(METRIC_ABORTED, 0)
            self.log_episode_score(METRIC_SUCCESS, 1 if score else 0)
            self.log_episode_score(METRIC_LOSE, 0 if score else 1)
            self.log_episode_score(BENCH_SCORE, 100)
        

class CloudgameBenchmark(GameBenchmark):
    """Integrate the game into the benchmark run."""
    def __init__(self):
        super().__init__(GAME_NAME)

    # defines whether the game is single player or not
    def is_single_player(self):
        return False

    # add a description of your game
    def get_description(self):
        return "A simple game in which a player has to decide whether they see clouds or not."

    # copy this, replacing the name of the game master in the return statement
    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> GameMaster:
        return Cloudgame(experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return CloudgameScorer(experiment, game_instance)