# Preparation for separating games and framework (whereby a benchmark run becomes a framework run with specific games)

## Preamble
### General Questions
* Naming confusion: The class `GameBenchmark` is used for a complete run of all instances of one game (not a set of specific games constituting a benchmark version)
* GameMaster vs. DialogueGameMaster: latter extends former/is the separation needed? former used in every game, the latter (additionally) in matchit/mapworld/hellogame/cloudgame/taboo, see example below:
```
class Taboo(DialogueGameMaster):
...

class TabooGameBenchmark(GameBenchmark):
    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> GameMaster:
        return Taboo(experiment, player_models)

```
* \__init\__.py in clemgame and backends defines project_root and configures logging, respectively. Is this redundant or am I missing something? Also, what's the difference in the loggers instantiated in chatgame/game.py and master.py?  

## TODOs:
* update test_benchmark.py (contains old versions of parameters and experiment names)
* adapt game development and instance generation to new setting
  * update import dependencies for games that use more than just master.py (such as game.py or utils; basically all but taboo and matchit...)
* test that scores stay the same 
* test that logging stays the same

## Preparational Thoughts
### Adding a new game
* implement game based on [template](#game-template)
* add entry in game registry specifying at least game name and game path

## Game Registry Fields:

```
{
"game_name": mandatory, game identifier, should be the same as GAME_NAME in master.py to avoid confusion (as the latter will be used for the results dierctory) (should we add a check?)
"game_path": mandatory, path to game  # absolute or relative to clemgame directory
"description": "A brief description of the game"
"main_game": "main game identifier" # to cluster different versions of the same game
"player": "single" | "two" | "multi"
"image": "none" | "single" | "multi"
"languages": ["en"] # list of ISO codes
"benchmark": ["X.X", "Y.Y"] # lists all benchmark versions in which this game was used 

# The games that are part of a specific collection can be filtered based on the 
# game attributes.
# For reproducibility, benchmark will also list all benchmark versions a game has   
# been used in previously.
}
```

## Isolate Game Template from Framework?
### (Abstract) Classes (clemgame/clemgame.py):
* InstanceGenerator
* Player
* GameMaster (DialogueGameMaster)?
* GameScorer?

### Game Structure
```
game
    in # directory containing instances_LANG_VERSION.json
    resources
        lang (optional)
            other resources
            initial_prompts
    instancegenerator.py # script reading in resources and generating instance file(s)
    game.py (sometimes also just master.py)
    master.py
```

### Results Structure
built by GameMaster and GameScorer, path specified as argument in cli.py, no changes needed

### possible game collections
* benchmark versions (currently different versions of code and instances, in the future only different instances)
  * text based benchmark (see clembench paper)
  * multimodal benchmark (see current version of the multimodal paper)
* game class (several versions of one game, for in-depth analysis)

### ChangeLog and Required Changes: 
```
clembench
   games # renamed to clemgame
   +--- __init__.py # contains 
   +--- custom_game_registry.json # added
   +--- game_registry.json # added
   clemgame # renamed to framework
  | 
  +--- __init__.py 
  |       --> add GameSpec class based on ModelSpec (from backends)
          --> load_custom_game_registry() and load_game_registry() similar to loading model registries
  +--- benchmark.py # renamed to framework.py
  |       list_games() # replaced to reading from game_registry
  |       run() # adapted to new game loading
  |       score() # adapted to new game loading
  |       transcripts() # adapted to new game loading
  +--- clemgame.py # moved to clemgame and renamed __init__.py
  |       GameResourceLocator # adapted to using game path instead of default location (for reading instances, but keep default for results)
  |       GameBenchmark - setup() # add game_path as argument
  |       load_benchmarks() # renamed to select_games() and adapted based on game registry and selection
  |       load_benchmark() # renamed to load_game() and adapted to load game from different location
  |       find_benchmark() # integrated into load_benchmark()
  +--- file_utils.py
  |       game_dir() # unlink from instance reading (in GameResourceLocator) but keep for keeping results structure the same
   scripts
  |
  +--- cli.py # renamed benchmark to framework
  |
   tests
  |
  +--- test_benchmark.py # rename to test_framework.py and adapt
  +--- logging.yaml # renamed main logger to clemcore.run
  +--- run_benchmark.sh # added to run a specific set of games constituting a benchmark version
  +--- game_selection.json # added to specify game properties for loading collections of games
```

