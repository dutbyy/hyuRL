
import gym
env = gym.make('CartPole-v1')
def CratpoleControl(p_x, i_x, d_x , ii,   episode, step_limit, isprint=True):
    tt = []
    for _ in range(episode):
        observation, _ = env.reset()
        for t in range(step_limit): 
            # env.render()
            x1, x2, x3, x4 = observation
            value = x1 * ii + x2 * i_x + x3 * p_x + x4 * d_x
            action = 1 if value > 0 else 0

            observation, reward, done, info, _ = env.step(action)  
            if done:
                if isprint:
                    print("Episode finished after {} timesteps".format(t+1))
                tt.append(t)
                break
        if isprint or t==step_limit:
            print("Episode finished after {} timesteps".format(t+1))
            tt.append(t) 
    return tt
if __name__ == '__main__': 
    CratpoleControl(0.1, 0, 0.005, 0.001,
                10, 10000,
                True)