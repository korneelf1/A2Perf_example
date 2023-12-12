import gin
import numpy as np

from rl_perf.domains.web_nav.gwob.CoDE import test_websites
from rl_perf.domains.web_nav.gwob.CoDE import utils
from rl_perf.domains.web_nav.gwob.CoDE import vocabulary_node
from rl_perf.domains.web_nav.gwob.CoDE import web_environment
from rl_perf.domains.web_nav.gwob.CoDE import web_primitives
from rl_perf.domains.web_nav.gwob.CoDE import q_networks

gin.parse_config_files_and_bindings(["/path/to/compositional_rl/gwob/configs/envdesign.gin"], None)

# Create an empty environment.
env = web_environment.GMiniWoBWebEnvironment(
  base_url="file:///path/to/compositional_rl/gwob/",
  global_vocabulary=vocabulary_node.LockedVocabulary())

# Create a q network.
q_net = q_networks.DQNWebLSTM(vocab_size=env.local_vocab.max_vocabulary_size, return_state_value=True)

# Sample a new design of the form {'number_of_pages': Integer, 'action': List[Integer], 'action_page': List[Integer]}.
# `action` denotes primitive indices and `action_page` denotes their corresponding page indices.
# Each item in the `action_page` should be less than `number_of_pages`.
# For this tutorial, we will randomly sample a design.
number_of_pages = np.random.randint(4) + 1
design =  {'number_of_pages': number_of_pages,
            'action': np.random.choice(np.arange(len(web_primitives.CONCEPTS)), 5),
            'action_page': np.random.choice(np.arange(number_of_pages), 5)}

# Design the actual environment.
env.design_environment(
    design, auto_num_pages=True)

# Reset the environment.
state = env.reset()

# Add batch dimension.
state = {key: np.expand_dims(tensor, axis=0) for key, tensor in state.items()}

# Get flattened logits and values.
logits, values = q_net(state)

# Get greedy action.
action = np.argmax(logits)

# Run the action.
new_state, reward, done, info = env.step(action)
