# In this Iteration

The reward functions were modified from:

```py
def get_adversary_reward(self):
    if self.reached_goal():
        return -self.success_reward
    return (self.ag_dist_to_target() - (self.dist_between_entities()/2))/16

def get_agent_reward(self):
    if self.reached_goal():
        return self.success_reward
```

to

```py
def get_adversary_reward(self):
    if self.reached_goal():
        return -self.success_reward
    # penalties are normalized 0 to -1 or 0 to 1
    dist_to_target_penalty = self.ag_dist_to_target() / (20 * 2**0.5) - 1
    dist_between_entities_penalty = -self.dist_between_entities() / (20 * 2**0.5)
    time_penalty = 1
    # penalties are multiplied by fractions of the success reward
    return (time_penalty * 0.7 + dist_between_entities_penalty * 0.3) / self.max_steps * self.success_reward

def get_agent_reward(self):
    if self.reached_goal():
        return self.success_reward
    
    # penalties are normalized 0 to -1
    dist_to_target_penalty = -self.ag_dist_to_target() / (20 * 2**0.5)
    time_penalty = -1
    # penalties are multiplied by fractions of the success reward
    return (time_penalty *1 + dist_to_target_penalty * 0.2) / self.max_steps * self.success_reward
```

The agents were turned into cylinders 👍

## Takeaways

1. This is bad. The model only experienced a few successes and it's memory is only 50 episodes
   1. maybe increase the memory by 900%? Increase batch size more?
2. Success might also be more sparse because of the new shape. Maybe the adversary is learning?
3. Maybe the higher rewards fuqed with the model
