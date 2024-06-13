"""Implement static and dynamic games with random parameters
   AUTHORS: Gabriel Vallat
"""
from typing import List, Tuple, Union
import torch as th
import torch.functional as F
import numpy as np
class StageGame():
    """A stage game with N players and n_a actions
    """
    def __init__(self,n_players,n_actions,coupling_w = 0, separable_type="additive", seed=None,device="cuda",aligned=True) -> None:
        """Initialise a stage game

        Args:
            n_players (int): Number of players
            n_actions (int): Number of actions for each players
            coupling_w (float, optional): coupling proportion. Defaults to 0.
            separable_type (Litteral, optional): Which mode is used to build the separable part of the reward"additive" or "multiplicative". Defaults to "additive".
            seed (int, optional): Seed of the random generator. Defaults to None.
            device (str, optional): Device to use. Defaults to "cuda".
            aligned (bool, optional): Whether both the separable and coupled reward have the same maximum. Defaults to True.
        """
        self.n = n_players
        self.na = n_actions
        self.seed = seed
        self.device=device
        self.gen = th.Generator(device=device)
        if seed is not None:
            self.gen.manual_seed(seed)
        assert coupling_w <= 1, "The proportional weight should sum to 1"

        self.r = th.zeros([self.na for i in range(self.n)],device=device)
        self.r_sep_indiv = th.rand(self.n,self.na,generator=self.gen,device=device)
        if separable_type == "additive":
            self.r_sep = self.r_sep_indiv[0].reshape([self.na]+[1]*(self.n-1))
            for p in range(1,self.n):
                shape = np.ones(self.n,dtype=int)
                shape[p]=self.na
                self.r_sep = self.r_sep+self.r_sep_indiv[p].reshape(tuple(shape))
        elif separable_type == "multiplicative":
            self.r_sep = self.r_sep_indiv[0].reshape([self.na]+[1]*(self.n-1))
            for p in range(1,self.n):
                shape = np.ones(self.n,dtype=int)
                shape[p]=self.na
                self.r_sep = self.r_sep*self.r_sep_indiv[p].reshape(tuple(shape))
        self.r_sep-=self.r_sep.min()
        self.r_sep/=self.r_sep.max()

        self.r_coupled = th.rand(self.n*[self.na],generator=self.gen,device=device)
        self.r_coupled-=self.r_coupled.min()
        self.r_coupled/=self.r_coupled.max()
        if aligned:
            argm_sep = np.unravel_index(th.argmax(self.r_sep).cpu(),self.r_sep.shape)
            r_coupled_swap = self.r_coupled[argm_sep]
            argm_coupled = np.unravel_index(th.argmax(self.r_coupled).cpu(),self.r_coupled.shape)
            self.r_coupled[argm_coupled]=r_coupled_swap
            self.r_coupled[argm_sep]=1
            
        self.r = self.r_coupled*coupling_w + self.r_sep*(1-coupling_w)
        #normalize the total reward:
        self.r-=self.r.min()
        self.r/=self.r.max()
    def play(self,actions: List[int])->int:
        return self.r[tuple(actions)]
    def solve_stoc_pol(self,stoc_pol):
        """Compute the expected reward for the given stochastic policy"""
        marginal_reward = self.r
        for p in range(self.n):
            shape = np.ones(self.n-p,dtype=int)
            shape[0] = self.na
            marginal_reward = (stoc_pol[p].reshape(tuple(shape))*marginal_reward).sum(0)
        return marginal_reward
    def solve_stoc_pol_batch(self,stoc_pol):
        """Compute the expected reward for each of the given stochastic policies"""
        marginal_reward = th.tile(self.r.unsqueeze(0),[stoc_pol.shape[0]]+self.n*[1])
        for p in range(self.n):
            shape = np.ones(1+self.n-p,dtype=int)
            shape[0] = stoc_pol.shape[0]
            shape[1] = self.na
            marginal_reward = (stoc_pol[:,p].reshape(tuple(shape))*marginal_reward).sum(1)
        return marginal_reward
    def is_nash(self,deterministic_pol):
        """Check that the deterministic policy given in input is a Nash equilibrium"""
        for p in range(self.n):
            pol_mp = list(deterministic_pol)
            pol_mp[p]=slice(None)
            if deterministic_pol[p]!=np.argmax(self.r.cpu().numpy()[tuple(pol_mp)]):
                return False
        return True
    def solve_greedy_pol_batch(self,stoc_pol):
        """Get the expected reward of the closest deterministic policy"""
        m,greedy_pol = stoc_pol.max(-1)
        greedy_pol_oh = th.nn.functional.one_hot(greedy_pol,num_classes=self.na)
        marginal_reward = th.tile(self.r.unsqueeze(0),[greedy_pol_oh.shape[0]]+self.n*[1])
        for p in range(self.n):
            shape = np.ones(1+self.n-p,dtype=int)
            shape[0] = greedy_pol_oh.shape[0]
            shape[1] = self.na
            marginal_reward = (greedy_pol_oh[:,p].reshape(tuple(shape))*marginal_reward).sum(1)
        return greedy_pol_oh,marginal_reward
class MultiStageGame():
    """A multi-stage random game, with ns states, n players and n actions
    """
    def __init__(self,n_players,n_states,
                 n_actions,
                 mult_r_w=0,
                 controlable_t_w=0,
                 discount_factor = 0.1,
                 seed=None,
                 device='cuda') -> None:
        self.n = n_players
        self.na = n_actions
        self.ns = n_states
        self.seed = seed
        self.random_np = np.random.default_rng(seed=seed)
        self.device=device
        self.gamma = 1-discount_factor
        self.gen = th.Generator(device=device)
        if seed is not None:
            self.gen.manual_seed(seed)
        assert mult_r_w <= 1, "The proportional weight of the cross terms in the reward function should be between 0 and 1"
        assert controlable_t_w <= 1,"The proportional weight of the cross terms in the transition function should be between 0 and 1"
        self.r = th.zeros([self.ns]+[self.na for i in range(self.n)],device=device)

        self.r_prod_indiv = th.rand([self.n]+[self.ns]+[self.na for i in range(self.n)],generator=self.gen,device=device)
        self.r_prod = self.r_prod_indiv.prod(0)
        self.r_prod = self.r_prod/self.r_prod.max()*mult_r_w #normalize to keep the resulting reward between 0 and mult_pot_w
        self.r += self.r_prod

        self.r_add_indiv = th.rand(self.n,self.ns,self.na,generator=self.gen,device=device)
        self.r_add = self.r_add_indiv[0].reshape([self.ns,self.na]+[1]*(self.n-1))
        for p in range(1,self.n):
            shape = np.ones(self.n+1,dtype=int)
            shape[0]=self.ns
            shape[p+1]=self.na
            self.r_add = self.r_add+self.r_add_indiv[p].reshape(tuple(shape))
        self.r_add = self.r_add/self.r_add.max()*(1-mult_r_w)
        self.r += self.r_add

        self.transition = th.zeros([self.ns]+[self.na for i in range(self.n)]+[self.ns],device = device)

        self.t_uncontrolable = th.rand([self.ns,self.ns],generator=self.gen,device=device)
        self.t_uncontrolable /= self.t_uncontrolable.sum(-1)[...,None]
        #same transition function whatever the actions are
        self.t_uncontrolable = th.tile(self.t_uncontrolable.reshape([self.ns]+[1 for i in range(self.n)]+[self.ns]),[1]+[self.na for i in range(self.n)]+[1])
        self.transition+=self.t_uncontrolable*(1-controlable_t_w)

        self.t_controlable = th.rand([self.ns]+[self.na for i in range(self.n)]+[self.ns],generator=self.gen,device=device)
        self.t_controlable /= self.t_controlable.sum(-1)[...,None]
           
        self.transition+=self.t_controlable*controlable_t_w

    def play_step(self,state,actions:List)->Tuple[int, int]:
        reward = self.r[tuple([state]+list(actions))]
        p = self.transition[tuple([state]+list(actions))].cpu().numpy()
        next_state = self.random_np.choice(np.arange(self.ns),p=p)
        return next_state,reward
    def play_game_CCE(self,central_policy,s_0=None):
        """Play the game using a central policy"""
        trans_pol = self.transition.reshape(self.ns,-1,self.ns)
        rew_pol = self.r.reshape(self.ns,-1)
        trans_pol = (trans_pol*central_policy.unsqueeze(-1)).sum(1)
        rew_pol = (rew_pol*central_policy).sum(1)
        #rew_pol = rew_pol.sum(dim=tuple((range(1,self.na))))
        #compute the discounted occupancy of each state
        disc_occ = th.linalg.inv(th.diag(th.ones(self.ns,device=self.device))-self.gamma*trans_pol)

        V_pol = disc_occ@rew_pol
        if s_0 is not None:
            return V_pol[s_0]
        else:
            return V_pol
    def play_game(self,policies,s_0:Union[int,None]=None):
        #first compute the induced Markov chain transition matrix and reward
        trans_pol = self.transition
        rew_pol = self.r
        for i in range(self.n):
            shape = np.ones(self.n+2-i,dtype=int)
            shape[0]=self.ns
            shape[1]=self.na
            trans_pol = trans_pol*policies[i].reshape(tuple(shape))
            trans_pol = trans_pol.sum(1)
            rew_pol = rew_pol*policies[i].reshape(tuple(shape[:-1]))
            rew_pol = rew_pol.sum(1)
        #rew_pol = rew_pol.sum(dim=tuple((range(1,self.na))))
        #compute the discounted occupancy of each state
        disc_occ = th.linalg.inv(th.diag(th.ones(self.ns,device=self.device))-self.gamma*trans_pol)

        V_pol = disc_occ@rew_pol
        if s_0 is not None:
            return V_pol[s_0]
        else:
            return V_pol
    def solve(self):
        """find the best deterministic policy"""
        pol = th.randint(0,self.na,size=(self.n,self.ns),device=self.device)
        while(True):
            V_pol = self.play_game(th.nn.functional.one_hot(pol,num_classes=self.na))
            Q_table = self.r+self.gamma*self.transition@V_pol
            old_pol = pol
            polidx = Q_table.reshape(self.ns,-1).argmax(axis=1)
            pol = th.tensor(np.array(np.unravel_index(polidx.cpu(),[self.na]*self.n)),device=self.device)
            if (old_pol==pol).all():
                break
        return th.nn.functional.one_hot(pol,num_classes=self.na),V_pol
def test_stage():
    game = StageGame(3,4,0,seed=3)
    action1 = [0,2,1]
    r1 = game.play(action1)
    print(f"Reward for action {action1}: {r1}")
    policy1 = th.zeros(game.n,game.na,device = game.device)
    policy1[th.arange(game.n),action1]=1
    e1 = game.solve_stoc_pol(policy1)
    print(f"Expected reward for the deterministic policy {action1}: {e1}")
    actions_stoc = np.tile(action1,(game.na,1))
    actions_stoc[np.arange(game.na),0]=np.arange(game.na)
    selected_player = 0
    stoc_pol = th.tensor([0.2,0.3,0.4,0.1])
    policy2 = policy1.clone()
    policy2[selected_player]=stoc_pol
    e2 = game.solve_stoc_pol(policy2)
    e3 = game.solve_stoc_pol_batch(th.stack([policy1,policy2]))
    a = action1
    r2=0
    for i in range(game.na):
        a[selected_player]=i
        r2+=stoc_pol[i]*game.play(a)
    print(f"Test passed: {e1==r1 and abs(e2-r2)<1e-5}")
def test_multi_stage():
    game = MultiStageGame(3,10,5,
                 mult_r_w=0,
                 controlable_t_w=0,
                 discount_factor = 0.1,
                 seed=1)
    policies = th.rand((game.n,game.ns,game.na),generator = game.gen,device = game.device)
    policies = policies/policies.sum(-1)[...,None]
    V_theoric = game.play_game(policies,s_0=0)
    #game.solve()
    V_emp = []
    for test in range(10):
        s = 0
        V_sample =0
        for i in range(10000):
            actions = [game.random_np.choice(np.arange(game.na),p = policies[k,s].cpu().numpy()) for k in range(game.n)]
            s,r = game.play_step(s,actions)
            V_sample += np.power(game.gamma,i)*r
        V_emp.append(V_sample.cpu().numpy())
    print(f"V theoric: {V_theoric}\nV empiric: {np.mean(V_emp)}")
if __name__ == "__main__":
    print("Start test")
    test_stage()